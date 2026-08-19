"""Utilities for transferable abstract orbital bases.

The continuation code stores transition RDMs in an abstract orthonormal AO
basis so they can be evaluated at other geometries.  Historically that basis
was the symmetric AO (SAO/OAO) Lowdin basis.  This module keeps SAO as the
default and adds meta-Lowdin and split-Procrustes alternatives.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from pyscf import gto, lo, scf


SAO_ALIASES = {"sao", "oao", "lowdin"}
META_LOWDIN_ALIASES = {"meta_lowdin", "meta-lowdin", "metalowdin"}
SPLIT_PROCRUSTES_ALIASES = {
    "split_procrustes",
    "canonical_split_procrustes",
    "canonical_split_procrustes_none",
    "canonical_split_procrustes_none_isolated",
}

DEFAULT_RHF_CONV_TOL = 1.0e-12
DEFAULT_RHF_CONV_TOL_GRAD = 1.0e-10
DEFAULT_RHF_CONV_TOL_CPSCF = 1.0e-10
DEFAULT_RHF_MAX_CYCLE = 100


@dataclass(frozen=True)
class SplitProcrustesInfo:
    """Rotations and overlaps used to build a split-Procrustes basis."""

    nocc: int
    occ_overlap: np.ndarray
    vrt_overlap: np.ndarray
    occ_rotation: np.ndarray
    vrt_rotation: np.ndarray


def normalize_basis_type(basis_type: str) -> str:
    """Return the canonical internal name for a basis-type string."""

    key = basis_type.lower().replace("-", "_")
    if key in SAO_ALIASES:
        return "SAO"
    if key in META_LOWDIN_ALIASES:
        return "meta_lowdin"
    if key in SPLIT_PROCRUSTES_ALIASES:
        return "split_procrustes"
    if key in {"canonical", "split"}:
        return key
    return key


def is_abstract_basis(basis_type: str) -> bool:
    """Whether ``basis_type`` is one of the packaged transfer bases."""

    return normalize_basis_type(basis_type) in {
        "SAO",
        "meta_lowdin",
        "split_procrustes",
    }


def basis_requires_reference(basis_type: str) -> bool:
    """Whether a basis needs a reference geometry/basis for smooth transfer."""

    return normalize_basis_type(basis_type) == "split_procrustes"


def get_loewdin_trafo(overlap_mat: np.ndarray, thresh: float = 1.0e-15) -> np.ndarray:
    """Compute the symmetric Lowdin transformation ``S^{-1/2}``."""

    vals, vecs = np.linalg.eigh(overlap_mat)
    inverse_sqrt_vals = np.where(vals > thresh, 1.0 / np.sqrt(vals), 0.0)
    return (vecs * inverse_sqrt_vals) @ vecs.conj().T


def _run_rhf(
    mol: gto.Mole,
    *,
    density_fit: bool = False,
    df_basis: str | None = None,
    conv_tol: float = DEFAULT_RHF_CONV_TOL,
    conv_tol_grad: float = DEFAULT_RHF_CONV_TOL_GRAD,
    conv_tol_cpscf: float = DEFAULT_RHF_CONV_TOL_CPSCF,
    max_cycle: int = DEFAULT_RHF_MAX_CYCLE,
) -> scf.hf.RHF:
    """Run a quiet RHF calculation and return the converged object."""

    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df_basis)
    mf.verbose = 0
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.conv_tol_cpscf = conv_tol_cpscf
    mf.max_cycle = max_cycle
    with contextlib.redirect_stdout(io.StringIO()):
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge while building an abstract basis")
    return mf


def _metalowdin_pre_orth_from_name(basis_type: str):
    name = basis_type.lower().replace("-", "_")
    if "none" in name:
        return None
    if "minao" in name:
        return "MINAO"
    if "scf" in name:
        return "SCF"
    return "ANO"


def get_meta_lowdin_basis(mol: gto.Mole, pre_orth_ao="ANO") -> np.ndarray:
    """Return PySCF's meta-Lowdin AO orthogonalization matrix."""

    return np.asarray(
        lo.orth.orth_ao(mol, method="meta_lowdin", pre_orth_ao=pre_orth_ao)
    )


def procrustes_rotation(overlap: np.ndarray, rcond: float = 1.0e-8) -> np.ndarray:
    """Return the orthogonal polar factor used for ``C_aligned = C @ R``."""

    u, singular_values, vt = np.linalg.svd(overlap, full_matrices=True)
    if singular_values.size == 0:
        return np.eye(overlap.shape[0], dtype=overlap.dtype)

    cutoff = rcond * singular_values[0]
    rank = np.count_nonzero(singular_values > cutoff)
    if rank == overlap.shape[0]:
        return u @ vt

    v = vt.T
    rotation = u[:, :rank] @ v[:, :rank].T

    u_null = u[:, rank:]
    v_null = v[:, rank:]
    u_fix, _, vt_fix = np.linalg.svd(u_null.T @ v_null)
    rotation += u_null @ (u_fix @ vt_fix) @ v_null.T
    return rotation


def split_procrustes_basis(
    mol: gto.Mole,
    basis_ref: np.ndarray | None = None,
    basis_ref_mol: gto.Mole | None = None,
    mf_object: scf.hf.RHF | None = None,
    ref_mf: scf.hf.RHF | None = None,
    procrustes_rcond: float = 1.0e-8,
    procrustes_overlap: str = "none",
    procrustes_density_fit: bool = False,
    procrustes_df_basis: str | None = None,
    procrustes_ref_density_fit: bool | None = None,
    procrustes_ref_df_basis: str | None = None,
    conv_tol: float = DEFAULT_RHF_CONV_TOL,
    conv_tol_grad: float = DEFAULT_RHF_CONV_TOL_GRAD,
    conv_tol_cpscf: float = DEFAULT_RHF_CONV_TOL_CPSCF,
    max_cycle: int = DEFAULT_RHF_MAX_CYCLE,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, SplitProcrustesInfo]:
    """Build a canonical occupied/virtual split-Procrustes basis.

    The currently implemented objective is the ``procrustes_overlap="none"``
    variant from the reference script: occupied and virtual canonical MO blocks
    are aligned independently using raw coefficient overlaps.
    """

    if procrustes_overlap.lower() != "none":
        raise NotImplementedError(
            "split_procrustes currently implements only procrustes_overlap='none'"
        )

    mf = mf_object if mf_object is not None else _run_rhf(
        mol,
        density_fit=procrustes_density_fit,
        df_basis=procrustes_df_basis,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_cpscf=conv_tol_cpscf,
        max_cycle=max_cycle,
    )

    ref_mol = basis_ref_mol
    if ref_mol is not None:
        from evcont.split_procrustes_derivatives import split_procrustes_basis_none

        ref_density_fit = (
            procrustes_density_fit
            if procrustes_ref_density_fit is None
            else procrustes_ref_density_fit
        )
        ref_df_basis = (
            procrustes_df_basis
            if procrustes_ref_df_basis is None
            else procrustes_ref_df_basis
        )
        if ref_mf is None:
            from evcont.split_procrustes_derivatives import run_rhf as run_procrustes_rhf

            ref_mf = run_procrustes_rhf(
                ref_mol,
                density_fit=ref_density_fit,
                df_basis=ref_df_basis,
                conv_tol=conv_tol,
                conv_tol_grad=conv_tol_grad,
                conv_tol_cpscf=conv_tol_cpscf,
                max_cycle=max_cycle,
            )

        return split_procrustes_basis_none(
            mol,
            ref_mol,
            mf=mf,
            ref_mf=ref_mf,
            rcond=procrustes_rcond,
            density_fit=procrustes_density_fit,
            df_basis=procrustes_df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
            return_info=return_info,
        )

    c = np.asarray(mf.mo_coeff)

    if basis_ref is None:
        basis = c
        if return_info:
            nocc = np.count_nonzero(np.asarray(mf.mo_occ) > 0)
            info = SplitProcrustesInfo(
                nocc=nocc,
                occ_overlap=np.eye(nocc),
                vrt_overlap=np.eye(c.shape[1] - nocc),
                occ_rotation=np.eye(nocc),
                vrt_rotation=np.eye(c.shape[1] - nocc),
            )
            return basis, info
        return basis

    basis_ref = np.asarray(basis_ref)
    if c.shape != basis_ref.shape:
        raise ValueError(
            "Current and reference basis matrices must have the same shape for "
            f"split-Procrustes, got {c.shape} and {basis_ref.shape}."
        )

    nocc = np.count_nonzero(np.asarray(mf.mo_occ) > 0)
    c_occ = c[:, :nocc]
    c_vrt = c[:, nocc:]
    c_ref_occ = basis_ref[:, :nocc]
    c_ref_vrt = basis_ref[:, nocc:]

    occ_overlap = c_occ.T @ c_ref_occ
    vrt_overlap = c_vrt.T @ c_ref_vrt

    occ_rotation = procrustes_rotation(occ_overlap, rcond=procrustes_rcond)
    vrt_rotation = procrustes_rotation(vrt_overlap, rcond=procrustes_rcond)

    basis = np.hstack((c_occ @ occ_rotation, c_vrt @ vrt_rotation))

    s = mol.intor_symmetric("int1e_ovlp")
    if not np.allclose(basis.T @ s @ basis, np.eye(basis.shape[1]), atol=1.0e-8):
        raise RuntimeError("split-Procrustes basis is not orthonormal")

    if not return_info:
        return basis

    info = SplitProcrustesInfo(
        nocc=nocc,
        occ_overlap=occ_overlap,
        vrt_overlap=vrt_overlap,
        occ_rotation=occ_rotation,
        vrt_rotation=vrt_rotation,
    )
    return basis, info


def get_basis_reference(
    mol: gto.Mole,
    basis_type: str = "SAO",
    mf_object: scf.hf.RHF | None = None,
    **kwargs,
) -> np.ndarray:
    """Build a reusable reference object for an abstract basis."""

    basis_name = normalize_basis_type(basis_type)
    if basis_name == "split_procrustes":
        return split_procrustes_basis(mol, mf_object=mf_object, basis_ref=None, **kwargs)
    return get_basis(mol, basis_type=basis_type, mf_object=mf_object, **kwargs)


def get_basis(
    mol: gto.Mole,
    basis_type: str = "SAO",
    basis_ref: np.ndarray | None = None,
    basis_ref_mol: gto.Mole | None = None,
    mf_object: scf.hf.RHF | None = None,
    ref_mf: scf.hf.RHF | None = None,
    procrustes_overlap: str = "none",
    procrustes_rcond: float = 1.0e-8,
    rcond: float | None = None,
    procrustes_density_fit: bool = False,
    procrustes_df_basis: str | None = None,
    procrustes_ref_density_fit: bool | None = None,
    procrustes_ref_df_basis: str | None = None,
    density_fit: bool | None = None,
    df_basis: str | None = None,
    conv_tol: float = DEFAULT_RHF_CONV_TOL,
    conv_tol_grad: float = DEFAULT_RHF_CONV_TOL_GRAD,
    conv_tol_cpscf: float = DEFAULT_RHF_CONV_TOL_CPSCF,
    max_cycle: int = DEFAULT_RHF_MAX_CYCLE,
    basis_orthonormality_atol: float = 1.0e-4,
    **kwargs,
) -> np.ndarray:
    """Construct an orthonormal AO-to-orbital coefficient matrix.

    Supported abstract transfer bases are ``"SAO"``/``"OAO"``,
    ``"meta_lowdin"``/``"metalowdin"``, and ``"split_procrustes"``.
    ``"canonical"`` and ``"split"`` are retained as computational-basis
    choices for solvers.
    """

    basis_name = normalize_basis_type(basis_type)
    s = mol.intor("int1e_ovlp")
    if rcond is not None:
        procrustes_rcond = rcond
    if density_fit is not None:
        procrustes_density_fit = density_fit
    if df_basis is not None:
        procrustes_df_basis = df_basis

    if basis_name == "SAO":
        basis = get_loewdin_trafo(s)

    elif basis_name == "meta_lowdin":
        basis = get_meta_lowdin_basis(
            mol, pre_orth_ao=_metalowdin_pre_orth_from_name(basis_type)
        )

    elif basis_name == "split_procrustes":
        basis = split_procrustes_basis(
            mol,
            basis_ref=basis_ref,
            basis_ref_mol=basis_ref_mol,
            mf_object=mf_object,
            ref_mf=ref_mf,
            procrustes_overlap=procrustes_overlap,
            procrustes_rcond=procrustes_rcond,
            procrustes_density_fit=procrustes_density_fit,
            procrustes_df_basis=procrustes_df_basis,
            procrustes_ref_density_fit=procrustes_ref_density_fit,
            procrustes_ref_df_basis=procrustes_ref_df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )

    elif basis_name in {"canonical", "split"}:
        mf = mf_object if mf_object is not None else _run_rhf(
            mol,
            density_fit=bool(density_fit),
            df_basis=df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
        basis = np.asarray(mf.mo_coeff)
        if basis_name == "split":
            nocc = np.count_nonzero(np.asarray(mf.mo_occ) > 0)
            localizer = lo.Boys(mol, basis[:, :nocc])
            localizer.init_guess = None
            basis_occ = localizer.kernel()
            localizer = lo.Boys(mol, basis[:, nocc:])
            localizer.init_guess = None
            basis_vrt = localizer.kernel()
            basis = np.concatenate((basis_occ, basis_vrt), axis=1)

    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")

    if basis.shape != (s.shape[0], s.shape[0]):
        raise ValueError(
            f"Basis {basis_type!r} has shape {basis.shape}, expected {(s.shape[0], s.shape[0])}"
        )
    orthogonality_error = np.max(
        np.abs(basis.T.conj() @ s @ basis - np.eye(basis.shape[1])),
        initial=0.0,
    )
    if orthogonality_error > basis_orthonormality_atol:
        raise RuntimeError(
            f"Basis {basis_type!r} is not orthonormal in the AO metric; "
            f"max error {orthogonality_error:.3e} exceeds "
            f"{basis_orthonormality_atol:.3e}"
        )

    return basis


def get_basis_with_derivative(
    mol: gto.Mole,
    basis_type: str = "SAO",
    basis_ref_mol: gto.Mole | None = None,
    mf_object: scf.hf.RHF | None = None,
    ref_mf: scf.hf.RHF | None = None,
    procrustes_rcond: float = 1.0e-8,
    procrustes_derivative_rcond: float = 1.0e-7,
    rcond: float | None = None,
    derivative_rcond: float | None = None,
    cphf_max_cycle: int = 50,
    cphf_level_shift: float = 0.0,
    procrustes_density_fit: bool = False,
    procrustes_df_basis: str | None = None,
    procrustes_ref_density_fit: bool | None = None,
    procrustes_ref_df_basis: str | None = None,
    density_fit: bool | None = None,
    df_basis: str | None = None,
    conv_tol: float = DEFAULT_RHF_CONV_TOL,
    conv_tol_grad: float = DEFAULT_RHF_CONV_TOL_GRAD,
    conv_tol_cpscf: float = DEFAULT_RHF_CONV_TOL_CPSCF,
    max_cycle: int = DEFAULT_RHF_MAX_CYCLE,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an abstract basis and its nuclear derivative.

    Derivatives are implemented for ``SAO``/``OAO``, ``meta_lowdin``, and
    ``split_procrustes``.
    """

    basis_name = normalize_basis_type(basis_type)
    if rcond is not None:
        procrustes_rcond = rcond
    if derivative_rcond is not None:
        procrustes_derivative_rcond = derivative_rcond
    if density_fit is not None:
        procrustes_density_fit = density_fit
    if df_basis is not None:
        procrustes_df_basis = df_basis

    if basis_name == "split_procrustes":
        if basis_ref_mol is None:
            raise ValueError(
                "basis_ref_mol must be supplied for split_procrustes derivatives"
            )
        from evcont.split_procrustes_derivatives import (
            run_rhf as run_procrustes_rhf,
            split_procrustes_basis_none,
            split_procrustes_basis_none_derivative,
        )

        ref_density_fit = (
            procrustes_density_fit
            if procrustes_ref_density_fit is None
            else procrustes_ref_density_fit
        )
        ref_df_basis = (
            procrustes_df_basis
            if procrustes_ref_df_basis is None
            else procrustes_ref_df_basis
        )
        if ref_mf is None:
            ref_mf = run_procrustes_rhf(
                basis_ref_mol,
                density_fit=ref_density_fit,
                df_basis=ref_df_basis,
                conv_tol=conv_tol,
                conv_tol_grad=conv_tol_grad,
                conv_tol_cpscf=conv_tol_cpscf,
                max_cycle=max_cycle,
            )

        basis = split_procrustes_basis_none(
            mol,
            basis_ref_mol,
            mf=mf_object,
            ref_mf=ref_mf,
            rcond=procrustes_rcond,
            density_fit=procrustes_density_fit,
            df_basis=procrustes_df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
        basis_grad = split_procrustes_basis_none_derivative(
            mol,
            basis_ref_mol,
            mf=mf_object,
            ref_mf=ref_mf,
            rcond=procrustes_rcond,
            derivative_rcond=procrustes_derivative_rcond,
            cphf_max_cycle=cphf_max_cycle,
            cphf_level_shift=cphf_level_shift,
            density_fit=procrustes_density_fit,
            df_basis=procrustes_df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
        return basis, basis_grad.transpose(2, 3, 0, 1)

    if basis_name not in {"SAO", "meta_lowdin"}:
        raise NotImplementedError(f"Basis derivatives are not implemented for {basis_type!r}")

    from evcont.localization_derivatives import orth_ao_derivative

    method = "lowdin" if basis_name == "SAO" else "meta_lowdin"
    pre_orth_ao = None if basis_name == "SAO" else _metalowdin_pre_orth_from_name(basis_type)
    return orth_ao_derivative(mol, method=method, pre_orth_ao=pre_orth_ao)
