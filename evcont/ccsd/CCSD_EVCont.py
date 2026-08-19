import numpy as np

from ebcc import REBCC
from ebcc.logging import NullLogger
from pyscf import ao2mo, lib, scf

from evcont.ccsd.RCCSD_rdm_mixed import make_rdm1_f, make_rdm2_f
from evcont.ab_initio_eigenvector_continuation import approximate_multistate
from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC
from evcont.basis.basis_utils import (
    basis_requires_reference,
    get_basis_reference,
    normalize_basis_type,
)
from evcont.electron_integral_utils import get_basis


def _run_rhf(
    mol,
    conv_tol=1.0e-12,
    conv_tol_grad=1.0e-10,
    conv_tol_cpscf=1.0e-10,
    max_cycle=100,
    density_fit=False,
    df_basis=None,
):
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df_basis)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.conv_tol_cpscf = conv_tol_cpscf
    mf.max_cycle = max_cycle
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def _run_ccsd_in_basis(mf, mo_coeff, solve_lambda=True, verbose=False):
    log = None if verbose else NullLogger()
    mf_basis = mf.copy()
    mf_basis.mo_coeff = mo_coeff
    ccsd = REBCC(mf_basis, ansatz="CCSD", log=log)
    ccsd.kernel()
    if solve_lambda:
        ccsd.solve_lambda()
    return ccsd


def _symmetrize_rdm1(rdm1):
    return 0.5 * (rdm1 + rdm1.conj().T)


def _symmetrize_rdm2(rdm2):
    return 0.5 * (rdm2 + np.einsum("ijkl->jilk", rdm2.conj()))


def _rdm_energy(mol, mf, mo_coeff, rdm1, rdm2):
    h1 = np.linalg.multi_dot((mo_coeff.T, mf.get_hcore(), mo_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), mol.nao)
    e_elec = lib.einsum("pq,qp->", h1, rdm1, optimize="optimal")
    e_elec += 0.5 * lib.einsum("pqrs,pqrs->", h2, rdm2, optimize="optimal")
    return float(np.real(e_elec + mol.energy_nuc()))


class CCSD_EVCont_obj:
    """Ground-state RCCSD eigenvector-continuation container.

    The object follows the SCI continuation convention of storing a reference
    geometry.  For AO-like abstract bases such as SAO and meta-Lowdin, CCSD
    amplitudes are solved in the reference computational MO gauge and mixed RDMs
    are transformed back to the abstract basis.  For reference-dependent
    Procrustes bases, the same reference geometry anchors the Procrustes
    alignment.
    """

    def __init__(
        self,
        comp_mol,
        nroots=1,
        abstract_basis="SAO",
        abstract_basis_ref=None,
        abstract_basis_ref_mol=None,
        abstract_basis_kwargs=None,
        hermitise="both",
        verbose_ccsd=False,
        scf_conv_tol=1.0e-12,
        scf_conv_tol_grad=1.0e-10,
        scf_conv_tol_cpscf=1.0e-10,
        scf_max_cycle=100,
        scf_density_fit=False,
        scf_df_basis=None,
    ):
        if comp_mol is None:
            raise ValueError("comp_mol must be provided for CCSD continuation")
        if nroots != 1:
            raise ValueError("CCSD continuation is currently implemented only for nroots=1")

        self.comp_mol = comp_mol
        self.nroots = nroots
        self.abstract_basis = abstract_basis
        self.abstract_basis_ref = abstract_basis_ref
        self.abstract_basis_ref_mol = abstract_basis_ref_mol
        self.abstract_basis_kwargs = dict(abstract_basis_kwargs or {})
        self.hermitise = hermitise
        self.verbose_ccsd = verbose_ccsd
        self.scf_conv_tol = scf_conv_tol
        self.scf_conv_tol_grad = scf_conv_tol_grad
        self.scf_conv_tol_cpscf = scf_conv_tol_cpscf
        self.scf_max_cycle = scf_max_cycle
        self.scf_density_fit = scf_density_fit
        self.scf_df_basis = scf_df_basis
        self._basis_name = normalize_basis_type(abstract_basis)

        if self._basis_name == "split_procrustes" and scf_density_fit:
            self.abstract_basis_kwargs.setdefault("procrustes_density_fit", True)
            self.abstract_basis_kwargs.setdefault("procrustes_df_basis", scf_df_basis)

        self._procrustes_density_fit = bool(
            self.abstract_basis_kwargs.get(
                "procrustes_density_fit",
                self.abstract_basis_kwargs.get("density_fit", False),
            )
        )
        self._procrustes_df_basis = self.abstract_basis_kwargs.get(
            "procrustes_df_basis",
            self.abstract_basis_kwargs.get("df_basis", None),
        )
        self._procrustes_ref_density_fit = self.abstract_basis_kwargs.get(
            "procrustes_ref_density_fit",
            self._procrustes_density_fit,
        )
        self._procrustes_ref_df_basis = self.abstract_basis_kwargs.get(
            "procrustes_ref_df_basis",
            self._procrustes_df_basis,
        )

        comp_density_fit = scf_density_fit or (
            self._basis_name == "split_procrustes" and self._procrustes_density_fit
        )
        comp_df_basis = scf_df_basis
        if self._basis_name == "split_procrustes" and self._procrustes_density_fit:
            comp_df_basis = self._procrustes_df_basis

        self.comp_mf = _run_rhf(
            comp_mol,
            conv_tol=scf_conv_tol,
            conv_tol_grad=scf_conv_tol_grad,
            conv_tol_cpscf=scf_conv_tol_cpscf,
            max_cycle=scf_max_cycle,
            density_fit=comp_density_fit,
            df_basis=comp_df_basis,
        )
        self.abstract_basis_ref_mf = self.comp_mf
        if (
            self._basis_name == "split_procrustes"
            and self._procrustes_ref_density_fit != comp_density_fit
        ) or (
            self._basis_name == "split_procrustes"
            and self._procrustes_ref_density_fit
            and self._procrustes_ref_df_basis != comp_df_basis
        ):
            self.abstract_basis_ref_mf = _run_rhf(
                comp_mol,
                conv_tol=scf_conv_tol,
                conv_tol_grad=scf_conv_tol_grad,
                conv_tol_cpscf=scf_conv_tol_cpscf,
                max_cycle=scf_max_cycle,
                density_fit=self._procrustes_ref_density_fit,
                df_basis=self._procrustes_ref_df_basis,
            )

        self._ensure_abstract_basis_reference(
            comp_mol,
            mf_object=self.abstract_basis_ref_mf,
        )

        self.comp_basis_abstract = self.get_abstract_basis(comp_mol, mf_object=self.comp_mf)
        self.comp_basis = get_basis(comp_mol, basis_type="canonical")
        s_comp = comp_mol.intor_symmetric("int1e_ovlp")
        self.global_trafo = np.einsum(
            "ji,jk,kl->il",
            self.comp_basis,
            s_comp,
            self.comp_basis_abstract,
            optimize="optimal",
        )
        self.use_computational_reference = self._basis_name in {
            "SAO",
            "meta_lowdin",
        }

        self.states = []
        self.mols = []
        self.ens = []
        self.ens_nuc = []
        self.mol_index = []
        self.train_energies = []

        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

    def _ensure_abstract_basis_reference(self, mol, mf_object=None):
        if not basis_requires_reference(self.abstract_basis):
            return
        if self.abstract_basis_ref is None:
            self.abstract_basis_ref = get_basis_reference(
                mol,
                basis_type=self.abstract_basis,
                mf_object=mf_object,
                **self.abstract_basis_kwargs,
            )
            self.abstract_basis_ref_mol = mol.copy()

    def get_abstract_basis(self, mol, mf_object=None):
        self._ensure_abstract_basis_reference(mol, mf_object=mf_object)
        return get_basis(
            mol,
            basis_type=self.abstract_basis,
            basis_ref=self.abstract_basis_ref,
            basis_ref_mol=self.abstract_basis_ref_mol,
            mf_object=mf_object,
            ref_mf=self.abstract_basis_ref_mf,
            **self.abstract_basis_kwargs,
        )

    def transformed_basis(self, mol, mf_object=None):
        """Return the orbital basis used to solve CCSD amplitudes."""
        basis_abstract = self.get_abstract_basis(mol, mf_object=mf_object)
        if self.use_computational_reference:
            return np.einsum("ij,kj->ik", basis_abstract, self.global_trafo, optimize="optimal")
        return basis_abstract

    def append_to_rdms(self, mol):
        """Append one ground-state CCSD training geometry and rebuild mixed RDMs."""
        mf = _run_rhf(
            mol,
            conv_tol=self.scf_conv_tol,
            conv_tol_grad=self.scf_conv_tol_grad,
            conv_tol_cpscf=self.scf_conv_tol_cpscf,
            max_cycle=self.scf_max_cycle,
            density_fit=self.scf_density_fit
            or (self._basis_name == "split_procrustes" and self._procrustes_density_fit),
            df_basis=self._procrustes_df_basis
            if self._basis_name == "split_procrustes" and self._procrustes_density_fit
            else self.scf_df_basis,
        )
        mo_coeff = self.transformed_basis(mol, mf_object=mf)
        ccsd = _run_ccsd_in_basis(
            mf,
            mo_coeff,
            solve_lambda=True,
            verbose=self.verbose_ccsd,
        )

        self.states.append(ccsd)
        self.mols.append(mol)
        self.ens.append(ccsd.e_tot)
        self.ens_nuc.append(mol.energy_nuc())
        self.mol_index.append(len(self.mols) - 1)
        self.train_energies.append(
            _rdm_energy(
                mol,
                mf,
                mo_coeff,
                ccsd.make_rdm1_f(hermitise=True),
                ccsd.make_rdm2_f(hermitise=True),
            )
        )
        self.build_transition_rdms()

    def build_transition_rdms(self):
        nstate = len(self.states)
        if nstate == 0:
            raise ValueError("No CCSD states have been appended")

        nao = self.comp_mol.nao
        nel = self.comp_mol.nelectron
        one_rdm = np.zeros((nstate, nstate, nao, nao))
        two_rdm = np.zeros((nstate, nstate, nao, nao, nao, nao))

        for j in range(nstate):
            for i in range(j, nstate):
                ccsd_i = self.states[i]
                ccsd_j = self.states[j]

                rdm1 = make_rdm1_f(
                    l1a=ccsd_i.l1,
                    l2a=ccsd_i.l2,
                    t1a=ccsd_i.t1,
                    t2a=ccsd_i.t2,
                    t1b=ccsd_j.t1,
                    t2b=ccsd_j.t2,
                )
                rdm2 = make_rdm2_f(
                    l1a=ccsd_i.l1,
                    l2a=ccsd_i.l2,
                    t1a=ccsd_i.t1,
                    t2a=ccsd_i.t2,
                    t1b=ccsd_j.t1,
                    t2b=ccsd_j.t2,
                )

                rdm1_conj = make_rdm1_f(
                    l1a=ccsd_j.l1,
                    l2a=ccsd_j.l2,
                    t1a=ccsd_j.t1,
                    t2a=ccsd_j.t2,
                    t1b=ccsd_i.t1,
                    t2b=ccsd_i.t2,
                )
                rdm2_conj = make_rdm2_f(
                    l1a=ccsd_j.l1,
                    l2a=ccsd_j.l2,
                    t1a=ccsd_j.t1,
                    t2a=ccsd_j.t2,
                    t1b=ccsd_i.t1,
                    t2b=ccsd_i.t2,
                )

                if self.hermitise == "both":
                    rdm1 = 0.5 * (rdm1 + rdm1_conj.conj().T)
                    rdm2 = 0.5 * (rdm2 + np.einsum("ijkl->jilk", rdm2_conj.conj()))
                elif self.hermitise != "none":
                    raise ValueError("hermitise must be 'both' or 'none'")

                rdm1 = _symmetrize_rdm1(rdm1)
                rdm2 = _symmetrize_rdm2(rdm2)

                if self.use_computational_reference:
                    rdm1 = np.einsum(
                        "...ij,ia,jb->...ab",
                        rdm1,
                        self.global_trafo,
                        self.global_trafo,
                        optimize="optimal",
                    )
                    rdm2 = np.einsum(
                        "...ijkl,ia,jb,kc,ld->...abcd",
                        rdm2,
                        self.global_trafo,
                        self.global_trafo,
                        self.global_trafo,
                        self.global_trafo,
                        optimize="optimal",
                    )

                one_rdm[i, j] = rdm1
                two_rdm[i, j] = rdm2
                one_rdm[j, i] = rdm1.conj().T
                two_rdm[j, i] = np.einsum("ijkl->jilk", rdm2.conj())

        self.one_rdm = one_rdm
        self.two_rdm = two_rdm
        self.overlap = np.einsum("abcc->ab", one_rdm, optimize="optimal") / nel

    def approximate(self, mol, nroots=1, hermitian=True):
        if nroots != 1:
            raise ValueError("CCSD continuation is currently implemented only for nroots=1")
        mf = _run_rhf(
            mol,
            conv_tol=self.scf_conv_tol,
            conv_tol_grad=self.scf_conv_tol_grad,
            conv_tol_cpscf=self.scf_conv_tol_cpscf,
            max_cycle=self.scf_max_cycle,
            density_fit=self.scf_density_fit
            or (self._basis_name == "split_procrustes" and self._procrustes_density_fit),
            df_basis=self._procrustes_df_basis
            if self._basis_name == "split_procrustes" and self._procrustes_density_fit
            else self.scf_df_basis,
        )
        basis = self.get_abstract_basis(mol, mf_object=mf)
        h1 = np.linalg.multi_dot((basis.T, mf.get_hcore(), basis))
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), mol.nao)
        energies, vec = approximate_multistate(
            h1,
            h2,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=1,
            hermitian=hermitian,
        )
        return np.real(energies + mol.energy_nuc()), vec

    def get_energy_with_grad(self, mol, nroots=1, **kwargs):
        if nroots != 1:
            raise ValueError("CCSD continuation is currently implemented only for nroots=1")
        basis_kwargs = dict(self.abstract_basis_kwargs)
        if self.abstract_basis_ref is not None:
            basis_kwargs.setdefault("basis_ref", self.abstract_basis_ref)
        if self._basis_name == "split_procrustes":
            basis_kwargs.setdefault("basis_ref_mol", self.abstract_basis_ref_mol)
            basis_kwargs.setdefault("ref_mf", self.abstract_basis_ref_mf)
        return get_multistate_energy_with_grad_and_NAC(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=1,
            abstract_basis=self.abstract_basis,
            basis_kwargs=basis_kwargs or None,
            **kwargs,
        )
