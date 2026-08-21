import warnings
from types import SimpleNamespace

import numpy as np

from ebcc import REBCC
from ebcc.logging import NullLogger
from pyscf import ao2mo, lib

from evcont.ccsd.RCCSD_rdm_mixed import make_rdm1_f, make_rdm2_f
from evcont.ab_initio_eigenvector_continuation import approximate_multistate
from evcont.basis.basis_utils import (
    AbstractBasisMixin,
    get_basis,
    normalize_basis_type,
    run_hf,
)
from evcont.solver_evaluation import EVContEvaluationMixin
from evcont.solver_persistence import EVContPersistenceMixin
from evcont.low_rank_utils import reduce_2rdm, vectorize_lowrank


_HF_BASED_ABSTRACT_BASES = {
    "canonical",
    "split",
    "split_procrustes",
    "least_change_atom_coordinate",
    "least_change_frozen_mo",
    "least_change_local",
}

_CONFIGURABLE_HF_ABSTRACT_BASES = {
    "split_procrustes",
    "least_change_atom_coordinate",
    "least_change_frozen_mo",
    "least_change_local",
}


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


def _mixed_rdms(bra, ket):
    bra_amplitudes = {
        "l1a": bra.l1,
        "l2a": bra.l2,
        "t1a": bra.t1,
        "t2a": bra.t2,
    }
    ket_amplitudes = {"t1b": ket.t1, "t2b": ket.t2}
    return (
        make_rdm1_f(**bra_amplitudes, **ket_amplitudes),
        make_rdm2_f(**bra_amplitudes, **ket_amplitudes),
    )


def _rdm_energy(mol, mf, mo_coeff, rdm1, rdm2):
    h1 = np.linalg.multi_dot((mo_coeff.T, mf.get_hcore(), mo_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), mol.nao)
    e_elec = lib.einsum("pq,qp->", h1, rdm1, optimize="optimal")
    e_elec += 0.5 * lib.einsum("pqrs,pqrs->", h2, rdm2, optimize="optimal")
    return float(np.real(e_elec + mol.energy_nuc()))


def _zero_amplitude_state(nocc, nvir, dtype=float):
    """Return the RCCSD amplitude representation of an RHF determinant."""

    return SimpleNamespace(
        t1=np.zeros((nocc, nvir), dtype=dtype),
        t2=np.zeros((nocc, nocc, nvir, nvir), dtype=dtype),
        l1=np.zeros((nvir, nocc), dtype=dtype),
        l2=np.zeros((nvir, nvir, nocc, nocc), dtype=dtype),
    )


class CCSD_EVCont_obj(
    AbstractBasisMixin, EVContEvaluationMixin, EVContPersistenceMixin
):
    """Ground-state RCCSD eigenvector-continuation container."""

    _lowrank_reduction_keys = {
        "truncation_style", "nvecs", "eval_thr", "ham_thr", "save_diag",
        "Jdiag_only", "relax_amp", "opt_no_diag", "relax_after", "use_svd",
        "svd_weight", "iterative", "nit", "max_iter_time", "min_eval",
    }

    def __init__(
        self,
        comp_mol,
        nroots=1,
        abstract_basis="split_procrustes",
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
        include_zero_amplitude=False,
        lowrank=False,
        lowrank_kwargs=None,
        **kwargs,
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
        self.include_zero_amplitude = include_zero_amplitude
        self.lowrank = lowrank
        self.kwargs = dict(lowrank_kwargs or {})
        self.kwargs.update(kwargs)
        self.lowrank_kwargs = {
            key: value for key, value in self.kwargs.items()
            if key in self._lowrank_reduction_keys
        }
        self.diagonal_lr = None
        self.vecs_lowrank = {}
        self._basis_name = normalize_basis_type(abstract_basis)
        self._reconcile_density_fitting()

        if (
            self.include_zero_amplitude
            and self._basis_name not in _HF_BASED_ABSTRACT_BASES
        ):
            raise ValueError(
                "include_zero_amplitude requires an RHF-based abstract basis "
                "that preserves the occupied/virtual split; use canonical, "
                "split, split_procrustes, or a least_change basis"
            )

        comp_hf_options = self._hf_options()
        self.comp_mf = run_hf(comp_mol, **comp_hf_options)
        self.abstract_basis_ref_mf = self.comp_mf
        ref_hf_options = self._hf_options(reference=True)
        if self._basis_name == "split_procrustes" and ref_hf_options != comp_hf_options:
            self.abstract_basis_ref_mf = run_hf(comp_mol, **ref_hf_options)

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
        self.zero_amplitude_state_index = None

        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

        if self.include_zero_amplitude:
            self._append_zero_amplitude_state()

    def _reconcile_density_fitting(self):
        """Use one consistent DF setting for CCSD and abstract-basis RHF."""

        if self._basis_name not in _CONFIGURABLE_HF_ABSTRACT_BASES:
            return

        basis_options = self.abstract_basis_kwargs
        if "density_fit" in basis_options:
            basis_density_fit = bool(basis_options["density_fit"])
        elif self._basis_name == "split_procrustes":
            basis_density_fit = bool(
                basis_options.get("procrustes_density_fit", False)
            )
        else:
            basis_density_fit = False

        if "df_basis" in basis_options:
            basis_df_is_explicit = True
            basis_df_basis = basis_options["df_basis"]
        elif (
            self._basis_name == "split_procrustes"
            and "procrustes_df_basis" in basis_options
        ):
            basis_df_is_explicit = True
            basis_df_basis = basis_options["procrustes_df_basis"]
        else:
            basis_df_is_explicit = False
            basis_df_basis = self.scf_df_basis

        if basis_density_fit:
            if not self.scf_density_fit:
                warnings.warn(
                    "abstract_basis_kwargs requests density fitting while "
                    "scf_density_fit=False; enabling density fitting for the "
                    "CCSD RHF calculations as well.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.scf_density_fit = True
            if basis_df_is_explicit and self.scf_df_basis != basis_df_basis:
                warnings.warn(
                    "abstract_basis_kwargs and scf_df_basis request different "
                    f"auxiliary bases; using {basis_df_basis!r} for the CCSD "
                    "RHF calculations as well.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.scf_df_basis = basis_df_basis

            # Generic names work for split-Procrustes and every least-change
            # basis, including later evaluation paths that build their own HF.
            basis_options["density_fit"] = True
            basis_options["df_basis"] = self.scf_df_basis
        elif self.scf_density_fit:
            # CCSD already uses DF, so record the same effective configuration
            # for any abstract-basis construction that does not receive its MF.
            basis_options["density_fit"] = True
            basis_options.setdefault("df_basis", self.scf_df_basis)

    def _hf_options(self, reference=False):
        options = {
            "conv_tol": self.scf_conv_tol,
            "conv_tol_grad": self.scf_conv_tol_grad,
            "conv_tol_cpscf": self.scf_conv_tol_cpscf,
            "max_cycle": self.scf_max_cycle,
            "density_fit": self.scf_density_fit,
            "df_basis": self.scf_df_basis,
        }
        if self._basis_name != "split_procrustes":
            return options

        density_fit = bool(
            self.abstract_basis_kwargs.get(
                "density_fit",
                self.abstract_basis_kwargs.get("procrustes_density_fit", False),
            )
        )
        df_basis = self.abstract_basis_kwargs.get(
            "df_basis",
            self.abstract_basis_kwargs.get("procrustes_df_basis"),
        )
        if reference:
            options["density_fit"] = self.abstract_basis_kwargs.get(
                "procrustes_ref_density_fit", density_fit
            )
            options["df_basis"] = self.abstract_basis_kwargs.get(
                "procrustes_ref_df_basis", df_basis
            )
        elif density_fit:
            options["density_fit"] = True
            options["df_basis"] = df_basis
        return options

    def _run_hf(self, mol):
        return run_hf(mol, **self._hf_options())

    def _append_zero_amplitude_state(self):
        """Add the geometry-independent RHF determinant to the subspace."""

        nocc = np.count_nonzero(np.asarray(self.comp_mf.mo_occ) > 0)
        nvir = self.comp_mf.mo_coeff.shape[1] - nocc
        state = _zero_amplitude_state(
            nocc,
            nvir,
            dtype=np.asarray(self.comp_mf.mo_coeff).dtype,
        )

        self.zero_amplitude_state_index = len(self.states)
        self.states.append(state)
        self.mols.append(self.comp_mol)
        self.ens.append(float(self.comp_mf.e_tot))
        self.ens_nuc.append(self.comp_mol.energy_nuc())
        self.mol_index.append(len(self.mols) - 1)
        self.train_energies.append(float(self.comp_mf.e_tot))
        self.build_transition_rdms()

    def transformed_basis(self, mol, mf_object=None):
        """Return the orbital basis used to solve CCSD amplitudes."""
        basis_abstract = self.get_abstract_basis(mol, mf_object=mf_object)
        if self.use_computational_reference:
            return np.einsum("ij,kj->ik", basis_abstract, self.global_trafo, optimize="optimal")
        return basis_abstract

    def append_to_rdms(self, mol):
        """Append one ground-state CCSD training geometry and rebuild mixed RDMs."""
        mf = self._run_hf(mol)
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
        old_nstate = 0 if self.one_rdm is None else self.one_rdm.shape[0]
        if old_nstate:
            one_rdm[:old_nstate, :old_nstate] = self.one_rdm
        two_rdm = None
        diagonal_lr = None
        vecs_lowrank = dict(self.vecs_lowrank)
        if not self.lowrank:
            two_rdm = np.zeros((nstate, nstate, nao, nao, nao, nao))
            if old_nstate:
                two_rdm[:old_nstate, :old_nstate] = self.two_rdm
        elif self.diagonal_lr is not None:
            diagonal_lr = np.zeros((nstate, nstate, 3, nao, nao))
            diagonal_lr[:old_nstate, :old_nstate] = self.diagonal_lr

        pairs = (
            [(i, j) for j in range(nstate) for i in range(j, nstate)]
            if old_nstate == 0
            else [(nstate - 1, j) for j in range(nstate)]
        )
        for i, j in pairs:
            ccsd_i = self.states[i]
            ccsd_j = self.states[j]

            rdm1, rdm2 = _mixed_rdms(ccsd_i, ccsd_j)
            rdm1_conj, rdm2_conj = _mixed_rdms(ccsd_j, ccsd_i)

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
            one_rdm[j, i] = rdm1.conj().T
            if self.lowrank:
                basis_kwargs = dict(self.abstract_basis_kwargs)
                if self.abstract_basis_ref_mol is not None:
                    basis_kwargs.setdefault(
                        "basis_ref_mol", self.abstract_basis_ref_mol
                    )
                if self._basis_name == "split_procrustes":
                    basis_kwargs.setdefault("ref_mf", self.abstract_basis_ref_mf)
                vectors, diagonals, joint = reduce_2rdm(
                    rdm1,
                    rdm2,
                    np.trace(rdm1) / nel,
                    mol=self.mols[i],
                    train_en=self.train_energies[i],
                    abstract_basis=self.abstract_basis,
                    basis_ref=self.abstract_basis_ref,
                    basis_kwargs=basis_kwargs,
                    **self.lowrank_kwargs,
                )
                values, left, right = vectors
                vecs_lowrank[i, j] = values, left, right, joint
                vecs_lowrank[j, i] = (
                    values.conj(),
                    left.conj(),
                    right.conj(),
                    joint,
                )
                if diagonals is not None:
                    if diagonal_lr is None:
                        diagonal_lr = np.zeros((nstate, nstate, 3, nao, nao))
                    diagonal_lr[i, j] = diagonals
                    diagonal_lr[j, i] = diagonals.conj()
            else:
                two_rdm[i, j] = rdm2
                two_rdm[j, i] = np.einsum("ijkl->jilk", rdm2.conj())

        self.one_rdm = one_rdm
        self.two_rdm = two_rdm
        self.diagonal_lr = diagonal_lr
        self.vecs_lowrank = vecs_lowrank
        self.lowrank_vectorized = None
        self.diagonal_vectorized = None
        self.overlap = np.einsum("abcc->ab", one_rdm, optimize="optimal") / nel

    def vectorize_lowrank(self, hermitian=True):
        vectorize_lowrank(self, hermitian=hermitian)

    def approximate(self, mol, nroots=1, hermitian=True):
        if nroots != 1:
            raise ValueError("CCSD continuation is currently implemented only for nroots=1")
        if self.lowrank:
            return self.get_en(mol, nroots=nroots, hermitian=hermitian)
        mf = self._run_hf(mol)
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
        """Compatibility alias for the original gradient-and-NAC evaluator."""
        return self.get_en_with_grad_and_NAC(mol, nroots=nroots, **kwargs)

    def _resolve_nroots(self, nroots):
        nroots = super()._resolve_nroots(nroots)
        if nroots != 1:
            raise ValueError("CCSD continuation is currently implemented only for nroots=1")
        return nroots

    def _persistence_state(self):
        state = dict(self.__dict__)
        state["states"] = [
            {name: np.asarray(getattr(ccsd, name)) for name in ("t1", "t2", "l1", "l2")}
            for ccsd in self.states
        ]
        return state

    def _restore_persistence_state(self):
        self.states = [SimpleNamespace(**state) for state in self.states]
