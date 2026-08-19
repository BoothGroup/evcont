#!/usr/bin/env python3
"""
Water CCSD continuation along a reaction coordinate in multiple abstract bases.

This follows the SCI continuation pattern: CCSD amplitudes for AO-like bases are
computed in a reference computational gauge, then mixed CCSD RDMs are transformed
back to the selected abstract representation for transferable ground-state
energy and gradient evaluation.  Direct PySCF CCSD energies and analytic
gradients are used as references.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

mpl_cache = Path(tempfile.gettempdir()) / "evcont-matplotlib"
xdg_cache = Path(tempfile.gettempdir()) / "evcont-xdg-cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ebcc import REBCC
from ebcc.logging import NullLogger
from pyscf import ao2mo, cc, gto, lib, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.RCCSD_rdm_mixed import make_rdm1_f, make_rdm2_f
from evcont.ab_initio_eigenvector_continuation import approximate_multistate
from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC
from evcont.basis_utils import (
    basis_requires_reference,
    get_basis,
    get_basis_reference,
    normalize_basis_type,
)


DIFFERENTIABLE_BASES = {"SAO", "meta_lowdin", "split_procrustes"}
ENERGY_ONLY_BASES = set()


def water_mol(d_oh=0.96, angle=104.5, asymmetry=0.0, basis="sto-3g"):
    theta = np.deg2rad(angle)
    half = 0.5 * theta
    d1 = d_oh * (1.0 + asymmetry)
    d2 = d_oh * (1.0 - asymmetry)
    atom = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, d1 * np.sin(half), -d1 * np.cos(half))),
        ("H", (0.0, -d2 * np.sin(half), -d2 * np.cos(half))),
    ]
    return gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)


def basis_label(basis_name):
    normalized = normalize_basis_type(basis_name)
    if normalized == "meta_lowdin":
        return "meta-Lowdin"
    if normalized == "split_procrustes":
        return "split-Procrustes"
    return basis_name


def run_rhf(mol, density_fit=False, df_basis=None):
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df_basis)
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-10
    mf.conv_tol_cpscf = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    return mf


def run_ccsd_in_basis(mol, mf, mo_coeff, solve_lambda=True, verbose=False):
    log = None if verbose else NullLogger()
    mf_basis = mf.copy()
    mf_basis.mo_coeff = mo_coeff
    ccsd = REBCC(mf_basis, ansatz="CCSD", log=log)
    ccsd.kernel()
    if solve_lambda:
        ccsd.solve_lambda()
    return ccsd


def symmetrize_rdm1(rdm1):
    return 0.5 * (rdm1 + rdm1.conj().T)


def symmetrize_rdm2(rdm2):
    return 0.5 * (rdm2 + np.einsum("ijkl->jilk", rdm2.conj()))


def rdm_energy(mol, mf, mo_coeff, rdm1, rdm2):
    h1 = np.linalg.multi_dot((mo_coeff.T, mf.get_hcore(), mo_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), mol.nao)
    e_elec = lib.einsum("pq,qp->", h1, rdm1, optimize="optimal")
    e_elec += 0.5 * lib.einsum("pqrs,pqrs->", h2, rdm2, optimize="optimal")
    return float(np.real(e_elec + mol.energy_nuc()))


def solve_continuation_energy(
    mol,
    rdm1,
    rdm2,
    overlap,
    abstract_basis,
    basis_ref,
    basis_ref_mol,
    ref_mf,
    basis_kwargs,
):
    use_df = normalize_basis_type(abstract_basis) == "split_procrustes" and bool(
        basis_kwargs.get("procrustes_density_fit", basis_kwargs.get("density_fit", False))
    )
    df_basis = basis_kwargs.get("procrustes_df_basis", basis_kwargs.get("df_basis", None))
    mf = run_rhf(mol, density_fit=use_df, df_basis=df_basis)
    basis = get_basis(
        mol,
        basis_type=abstract_basis,
        basis_ref=basis_ref,
        basis_ref_mol=basis_ref_mol,
        mf_object=mf,
        ref_mf=ref_mf,
        **basis_kwargs,
    )
    h1 = np.linalg.multi_dot((basis.T, mf.get_hcore(), basis))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), mol.nao)
    energies, vec = approximate_multistate(h1, h2, rdm1, rdm2, overlap, nroots=1)
    return np.real(energies + mol.energy_nuc()), vec


class CCSDContinuation:
    def __init__(self, comp_mol, abstract_basis, basis_kwargs=None, verbose_ccsd=False):
        self.comp_mol = comp_mol
        self.abstract_basis = abstract_basis
        self.basis_kwargs = dict(basis_kwargs or {})
        self.verbose_ccsd = verbose_ccsd
        self.basis_ref = None
        self.basis_ref_mol = None
        self.basis_ref_mf = None
        self.use_procrustes_density_fit = normalize_basis_type(abstract_basis) == "split_procrustes" and bool(
            self.basis_kwargs.get(
                "procrustes_density_fit",
                self.basis_kwargs.get("density_fit", False),
            )
        )
        self.procrustes_df_basis = self.basis_kwargs.get(
            "procrustes_df_basis",
            self.basis_kwargs.get("df_basis", None),
        )
        self.use_procrustes_ref_density_fit = self.basis_kwargs.get(
            "procrustes_ref_density_fit",
            self.use_procrustes_density_fit,
        )
        self.procrustes_ref_df_basis = self.basis_kwargs.get(
            "procrustes_ref_df_basis",
            self.procrustes_df_basis,
        )

        self.comp_mf = run_rhf(
            comp_mol,
            density_fit=self.use_procrustes_density_fit,
            df_basis=self.procrustes_df_basis,
        )
        self.basis_ref_mf = self.comp_mf
        if (
            normalize_basis_type(abstract_basis) == "split_procrustes"
            and self.use_procrustes_ref_density_fit != self.use_procrustes_density_fit
        ) or (
            normalize_basis_type(abstract_basis) == "split_procrustes"
            and self.use_procrustes_ref_density_fit
            and self.procrustes_ref_df_basis != self.procrustes_df_basis
        ):
            self.basis_ref_mf = run_rhf(
                comp_mol,
                density_fit=self.use_procrustes_ref_density_fit,
                df_basis=self.procrustes_ref_df_basis,
            )
        if basis_requires_reference(abstract_basis):
            self.basis_ref = get_basis_reference(
                comp_mol,
                basis_type=abstract_basis,
                mf_object=self.basis_ref_mf,
                **self.basis_kwargs,
            )
            self.basis_ref_mol = comp_mol.copy()

        self.comp_basis = get_basis(
            comp_mol,
            basis_type=abstract_basis,
            basis_ref=self.basis_ref,
            basis_ref_mol=self.basis_ref_mol,
            mf_object=self.comp_mf,
            ref_mf=self.basis_ref_mf,
            **self.basis_kwargs,
        )
        s_comp = comp_mol.intor_symmetric("int1e_ovlp")
        self.comp_to_abstract = np.einsum(
            "ji,jk,kl->il",
            self.comp_mf.mo_coeff,
            s_comp,
            self.comp_basis,
            optimize="optimal",
        )
        self.use_comp_basis = normalize_basis_type(abstract_basis) in {"SAO", "meta_lowdin"}
        self.states = []
        self.train_energies = []
        self.one_rdm = None
        self.two_rdm = None
        self.overlap = None

    def abstract_basis_for_mol(self, mol, mf):
        return get_basis(
            mol,
            basis_type=self.abstract_basis,
            basis_ref=self.basis_ref,
            basis_ref_mol=self.basis_ref_mol,
            mf_object=mf,
            ref_mf=self.basis_ref_mf,
            **self.basis_kwargs,
        )

    def computational_basis_for_mol(self, mol, mf):
        basis = self.abstract_basis_for_mol(mol, mf)
        if self.use_comp_basis:
            return np.einsum("ij,kj->ik", basis, self.comp_to_abstract, optimize="optimal")
        return basis

    def train(self, mols):
        for mol in mols:
            mf = run_rhf(
                mol,
                density_fit=self.use_procrustes_density_fit,
                df_basis=self.procrustes_df_basis,
            )
            mo_coeff = self.computational_basis_for_mol(mol, mf)
            ccsd = run_ccsd_in_basis(mol, mf, mo_coeff, verbose=self.verbose_ccsd)
            self.states.append(ccsd)
            self.train_energies.append(
                rdm_energy(
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
                rdm1 = 0.5 * (rdm1 + rdm1_conj.conj().T)
                rdm2 = 0.5 * (rdm2 + np.einsum("ijkl->jilk", rdm2_conj.conj()))
                rdm1 = symmetrize_rdm1(rdm1)
                rdm2 = symmetrize_rdm2(rdm2)

                if self.use_comp_basis:
                    rdm1 = np.einsum(
                        "...ij,ia,jb->...ab",
                        rdm1,
                        self.comp_to_abstract,
                        self.comp_to_abstract,
                        optimize="optimal",
                    )
                    rdm2 = np.einsum(
                        "...ijkl,ia,jb,kc,ld->...abcd",
                        rdm2,
                        self.comp_to_abstract,
                        self.comp_to_abstract,
                        self.comp_to_abstract,
                        self.comp_to_abstract,
                        optimize="optimal",
                    )

                one_rdm[i, j] = rdm1
                two_rdm[i, j] = rdm2
                one_rdm[j, i] = rdm1.conj().T
                two_rdm[j, i] = np.einsum("ijkl->jilk", rdm2.conj())

        self.one_rdm = one_rdm
        self.two_rdm = two_rdm
        self.overlap = np.einsum("abcc->ab", one_rdm, optimize="optimal") / nel

    def continuation_energy(self, mol, nroots):
        energies, vec = solve_continuation_energy(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            self.abstract_basis,
            self.basis_ref,
            self.basis_ref_mol,
            self.basis_ref_mf,
            self.basis_kwargs,
        )
        return energies[:nroots], vec[:nroots]

    def continuation_properties(self, mol, nroots):
        basis_kwargs = dict(self.basis_kwargs)
        if normalize_basis_type(self.abstract_basis) == "split_procrustes":
            basis_kwargs.setdefault("basis_ref_mol", self.basis_ref_mol)
            basis_kwargs.setdefault("ref_mf", self.basis_ref_mf)
        return get_multistate_energy_with_grad_and_NAC(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=nroots,
            abstract_basis=self.abstract_basis,
            basis_kwargs=basis_kwargs or None,
        )


def coordinate_values(args):
    if args.coord == "angle":
        train_values = np.asarray(args.train_angles, dtype=float)
        test_values = np.linspace(args.angle_min, args.angle_max, args.test_points)
        x_label = r"H-O-H angle ($^\circ$)"
    else:
        train_values = np.asarray(args.train_bonds, dtype=float)
        test_values = np.linspace(args.bond_min, args.bond_max, args.test_points)
        x_label = r"O-H distance ($\AA$)"
    return train_values, test_values, x_label


def mol_from_coordinate(value, args):
    if args.coord == "angle":
        return water_mol(args.bond, value, asymmetry=args.asymmetry, basis=args.basis)
    return water_mol(value, args.angle, asymmetry=args.asymmetry, basis=args.basis)


def comp_mol_from_args(args):
    return water_mol(
        args.comp_bond,
        args.comp_angle,
        asymmetry=args.asymmetry,
        basis=args.basis,
    )


def direct_ccsd_ground_reference(test_values, args):
    energies = np.zeros(len(test_values))
    gradients = []
    abs_grad = np.zeros(len(test_values))
    for i, value in enumerate(test_values):
        mol = mol_from_coordinate(value, args)
        mf = run_rhf(mol)
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1.0e-8
        mycc.kernel()
        grad_i = mycc.nuc_grad_method().kernel()
        energies[i] = mycc.e_tot
        gradients.append(grad_i)
        abs_grad[i] = np.linalg.norm(grad_i)
    return {
        "energies": energies,
        "grad": np.asarray(gradients),
        "abs_grad": abs_grad,
    }


def evaluate_continuation(cont, test_values, args, include_properties):
    energies = np.zeros(len(test_values))
    abs_grad = np.full(len(test_values), np.nan)
    grad_all = []

    for i, value in enumerate(test_values):
        mol = mol_from_coordinate(value, args)
        if include_properties:
            _, en_i, grad_i, _, _ = cont.continuation_properties(mol, 1)
            energies[i] = en_i[0]
            grad_all.append(grad_i[0])
            abs_grad[i] = np.linalg.norm(grad_i[0])
        else:
            en_i, _ = cont.continuation_energy(mol, 1)
            energies[i] = en_i[0]

    return {
        "energies": energies,
        "abs_grad": abs_grad,
        "grad": np.asarray(grad_all),
        "train_energies": np.asarray(cont.train_energies),
    }


def plot_results(test_values, train_values, x_label, reference, data_by_basis, out_path):
    basis_styles = {
        "SAO": {"color": "#006EAF", "ls": "--"},
        "meta_lowdin": {"color": "#D24000", "ls": ":"},
        "split_procrustes": {"color": "#426A5A", "ls": "-."},
    }
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        figsize=(8.4, 6.2),
        gridspec_kw={"hspace": 0.08, "wspace": 0.30},
        height_ratios=[1.5, 1.0],
    )

    axes[0][0].plot(
        test_values,
        reference["energies"],
        color="#222222",
        alpha=0.35,
        lw=2,
        label="direct CCSD",
    )
    axes[0][1].plot(
        test_values,
        reference["abs_grad"],
        color="#222222",
        alpha=0.35,
        lw=2,
        label="direct CCSD",
    )

    for basis_name, data in data_by_basis.items():
        normalized = normalize_basis_type(basis_name)
        style = basis_styles.get(normalized, {"color": "#4B5563", "ls": "-."})
        label_basis = basis_label(basis_name)
        axes[0][0].plot(
            test_values,
            data["energies"],
            color=style["color"],
            ls=style["ls"],
            lw=1.6,
            alpha=0.8,
            label=label_basis,
        )
        axes[1][0].plot(
            test_values,
            np.abs(data["energies"] - reference["energies"]),
            color=style["color"],
            ls=style["ls"],
            lw=1.4,
            alpha=0.8,
        )

        if normalized in ENERGY_ONLY_BASES:
            continue

        axes[0][1].plot(
            test_values,
            data["abs_grad"],
            color=style["color"],
            ls=style["ls"],
            lw=1.6,
            alpha=0.78,
            label=label_basis,
        )
        force_error = np.linalg.norm(data["grad"] - reference["grad"], axis=(1, 2))
        axes[1][1].plot(
            test_values,
            force_error,
            color=style["color"],
            ls=style["ls"],
            lw=1.4,
            alpha=0.78,
            label=label_basis,
        )

    for ax in axes[1]:
        ax.set_yscale("log")

    for row in axes:
        for ax in row:
            ylims = ax.get_ylim()
            ax.vlines(train_values, ymin=ylims[0], ymax=ylims[1], ls="--", color="gray", alpha=0.45)
            ax.set_ylim(ylims)
            ax.grid(ls=":", alpha=0.6)

    axes[0][0].set_title("Energies")
    axes[0][1].set_title("Force Norms")

    axes[0][0].set_ylabel("Energy (Ha)")
    axes[0][1].set_ylabel(r"$||F||$ (Ha $a_0^{-1}$)")
    axes[1][0].set_ylabel(r"$|E_\mathrm{CCSD} - E_\mathrm{cont}|$ (Ha)")
    axes[1][1].set_ylabel(r"$||F_\mathrm{CCSD} - F_\mathrm{cont}||$ (Ha $a_0^{-1}$)")

    fig.text(0.5, 0.02, x_label, ha="center")
    for ax, fontsize in [
        (axes[0][0], 8),
        (axes[0][1], 7),
        (axes[1][1], 7),
    ]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=fontsize)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="cc-pvdz", help="PySCF AO basis")
    parser.add_argument("--coord", choices=["angle", "bond"], default="angle")
    parser.add_argument("--bond", type=float, default=0.96, help="Fixed O-H bond in Angstrom")
    parser.add_argument("--angle", type=float, default=104.5, help="Fixed H-O-H angle in degrees")
    parser.add_argument("--asymmetry", type=float, default=0.0)
    parser.add_argument("--comp-bond", type=float, default=1.05)
    parser.add_argument("--comp-angle", type=float, default=120.0)
    parser.add_argument("--train-angles", type=float, nargs="+", default=[70.0, 104.5, 140.0])
    parser.add_argument("--train-bonds", type=float, nargs="+", default=[0.85, 0.96, 1.12])
    parser.add_argument("--angle-min", type=float, default=55.0)
    parser.add_argument("--angle-max", type=float, default=160.0)
    parser.add_argument("--bond-min", type=float, default=0.78)
    parser.add_argument("--bond-max", type=float, default=1.25)
    parser.add_argument("--test-points", type=int, default=21)
    parser.add_argument("--nroots", type=int, default=1)
    parser.add_argument(
        "--abstract-bases",
        nargs="+",
        default=["SAO", "meta_lowdin", "split_procrustes"],
    )
    parser.add_argument(
        "--procrustes-density-fit",
        action="store_true",
        help="Use density-fitted RHF objects when constructing split-Procrustes bases.",
    )
    parser.add_argument(
        "--procrustes-ref-density-fit",
        action="store_true",
        help="Use a density-fitted RHF object for only the split-Procrustes reference.",
    )
    parser.add_argument("--procrustes-df-basis", default=None)
    parser.add_argument("--procrustes-ref-df-basis", default=None)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("water_ccsd_energy_gradient_basis_comparison.png")),
    )
    parser.add_argument("--quick", action="store_true", help="Use a tiny scan for smoke testing")
    parser.add_argument("--verbose-ccsd", action="store_true", help="Show EBCC iteration output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.basis = "sto-3g"
        args.test_points = 5
        args.train_angles = [80.0, 110.0, 140.0]
        args.train_bonds = [0.86, 0.96, 1.10]
    if args.nroots != 1:
        raise ValueError("CCSD continuation is currently implemented only for nroots=1")

    train_values, test_values, x_label = coordinate_values(args)
    train_mols = [mol_from_coordinate(value, args) for value in train_values]
    comp_mol = comp_mol_from_args(args)

    print("Computing direct ground-state CCSD energy/gradient reference", flush=True)
    reference = direct_ccsd_ground_reference(test_values, args)

    data_by_basis = {}
    for abstract_basis in args.abstract_bases:
        print(f"Training CCSD continuation in {abstract_basis}", flush=True)
        basis_kwargs = {"procrustes_overlap": "none"}
        if normalize_basis_type(abstract_basis) == "split_procrustes":
            basis_kwargs.update(
                {
                    "procrustes_density_fit": args.procrustes_density_fit,
                    "procrustes_ref_density_fit": args.procrustes_ref_density_fit
                    or args.procrustes_density_fit,
                    "procrustes_df_basis": args.procrustes_df_basis,
                    "procrustes_ref_df_basis": args.procrustes_ref_df_basis
                    or args.procrustes_df_basis,
                }
            )
        cont = CCSDContinuation(
            comp_mol,
            abstract_basis,
            basis_kwargs=basis_kwargs,
            verbose_ccsd=args.verbose_ccsd,
        )
        cont.train(train_mols)
        include_properties = normalize_basis_type(abstract_basis) in DIFFERENTIABLE_BASES
        print(f"Evaluating {abstract_basis}", flush=True)
        data_by_basis[abstract_basis] = evaluate_continuation(
            cont,
            test_values,
            args,
            include_properties=include_properties,
        )

    out_path = Path(args.out).resolve()
    plot_results(test_values, train_values, x_label, reference, data_by_basis, out_path)
    print(f"Saved plot to {out_path}", flush=True)


if __name__ == "__main__":
    main()
