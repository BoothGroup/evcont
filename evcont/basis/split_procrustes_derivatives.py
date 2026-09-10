#!/usr/bin/env python3
"""Split-Procrustes-none orbital bases and their analytic nuclear derivatives.

The occupied and virtual RHF spaces are aligned independently to fixed
reference coefficient arrays.  The Procrustes objectives use ordinary
Euclidean products of AO coefficient arrays; no same- or cross-geometry AO
overlap enters the objective.

The analytic derivative combines PySCF's occupied RHF CPHF response with a
symmetric-gauge completion of the virtual response and a direct derivative of
the orthogonal polar factor.  Only real molecular RHF calculations with a
full-rank occupied and virtual matching matrix are supported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hessian

from evcont.basis.basis_utils import run_hf


class ProcrustesDerivativeError(ValueError):
    """Raised when the requested polar-factor derivative is not well defined."""


@dataclass(frozen=True)
class SplitProcrustesInfo:
    """Rotations and matching matrices used to build the aligned basis."""

    nocc: int
    occ_overlap: np.ndarray
    vrt_overlap: np.ndarray
    occ_rotation: np.ndarray
    vrt_rotation: np.ndarray


@dataclass(frozen=True)
class RHFMOResponseInfo:
    """Intermediate RHF response quantities in the chosen orbital gauge."""

    atom_indices: tuple[int, ...]
    overlap_derivative: np.ndarray
    mo_response: np.ndarray
    occupied_coefficient_derivative: np.ndarray
    orbital_energy_derivative: np.ndarray


@dataclass(frozen=True)
class SplitProcrustesDerivativeInfo:
    """Intermediate quantities used for the split-basis derivative."""

    atom_indices: tuple[int, ...]
    nocc: int
    overlap_derivative: np.ndarray
    mo_coefficient_derivative: np.ndarray
    mo_response: np.ndarray
    occ_overlap: np.ndarray
    vrt_overlap: np.ndarray
    occ_rotation: np.ndarray
    vrt_rotation: np.ndarray
    occ_singular_values: np.ndarray
    vrt_singular_values: np.ndarray
    occ_overlap_derivative: np.ndarray
    vrt_overlap_derivative: np.ndarray
    occ_rotation_derivative: np.ndarray
    vrt_rotation_derivative: np.ndarray


def _validate_square_real(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _occupation_count(mf: scf.hf.RHF, name: str) -> int:
    mo_occ = np.asarray(mf.mo_occ)
    nocc = int(np.count_nonzero(mo_occ > 0))
    expected = np.zeros_like(mo_occ, dtype=float)
    expected[:nocc] = 2.0
    if not np.allclose(mo_occ, expected):
        raise ValueError(
            f"{name} must have contiguous doubly occupied RHF orbitals followed "
            f"by virtual orbitals; got occupations {mo_occ}."
        )
    return nocc


def _validate_rhf(mf: scf.hf.RHF, name: str) -> None:
    if not isinstance(mf, scf.hf.RHF) or hasattr(mf, "xc"):
        raise TypeError(f"{name} must be a molecular RHF calculation")
    if not mf.converged:
        raise RuntimeError(f"{name} is not converged")
    for attribute in ("mo_coeff", "mo_energy", "mo_occ"):
        value = np.asarray(getattr(mf, attribute))
        if np.iscomplexobj(value):
            raise TypeError(f"Complex-valued {name}.{attribute} is not supported")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}.{attribute} contains non-finite values")
    _occupation_count(mf, name)


def _validate_same_geometry(mol: gto.Mole, mf: scf.hf.RHF, name: str) -> None:
    mf_mol = mf.mol
    if mol.natm != mf_mol.natm:
        raise ValueError(f"{name} does not correspond to the supplied molecule")
    if not np.array_equal(mol.atom_charges(), mf_mol.atom_charges()):
        raise ValueError(f"{name} has different atoms from the supplied molecule")
    if not np.allclose(mol.atom_coords(), mf_mol.atom_coords(), atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} was evaluated at a different geometry")


def _validate_compatible_ao_bases(mol: gto.Mole, ref_mol: gto.Mole) -> None:
    if mol.natm != ref_mol.natm:
        raise ValueError("Current and reference molecules have different atom counts")
    if tuple(mol.atom_pure_symbol(i) for i in range(mol.natm)) != tuple(
        ref_mol.atom_pure_symbol(i) for i in range(ref_mol.natm)
    ):
        raise ValueError("Current and reference atom ordering differs")
    if bool(mol.cart) != bool(ref_mol.cart):
        raise ValueError("Current and reference Cartesian/spherical AO conventions differ")
    if mol.nao_nr() != ref_mol.nao_nr() or mol.nbas != ref_mol.nbas:
        raise ValueError("Current and reference AO dimensions differ")
    if tuple(mol.ao_labels()) != tuple(ref_mol.ao_labels()):
        raise ValueError("Current and reference AO labels or ordering differ")

    shell_fields = ("bas_atom", "bas_angular", "bas_nprim", "bas_nctr", "bas_kappa")
    for shell in range(mol.nbas):
        for field in shell_fields:
            if getattr(mol, field)(shell) != getattr(ref_mol, field)(shell):
                raise ValueError(f"Current and reference basis shells differ at shell {shell}")
        if not np.allclose(mol.bas_exp(shell), ref_mol.bas_exp(shell), rtol=0.0, atol=0.0):
            raise ValueError(f"Current and reference basis exponents differ at shell {shell}")
        if not np.allclose(
            mol.bas_ctr_coeff(shell),
            ref_mol.bas_ctr_coeff(shell),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"Current and reference contraction coefficients differ at shell {shell}"
            )


def _validate_calculations(
    mol: gto.Mole,
    ref_mol: gto.Mole,
    mf: scf.hf.RHF,
    ref_mf: scf.hf.RHF,
) -> int:
    _validate_rhf(mf, "mf")
    _validate_rhf(ref_mf, "ref_mf")
    _validate_same_geometry(mol, mf, "mf")
    _validate_same_geometry(ref_mol, ref_mf, "ref_mf")
    _validate_compatible_ao_bases(mol, ref_mol)

    c = np.asarray(mf.mo_coeff)
    c_ref = np.asarray(ref_mf.mo_coeff)
    if c.shape != c_ref.shape:
        raise ValueError(
            "Current and reference MO coefficient matrices must have the same "
            f"shape, got {c.shape} and {c_ref.shape}."
        )
    if c.shape[0] != c.shape[1]:
        raise ValueError(
            "A complete square molecular AO-to-MO transformation is required; "
            f"got shape {c.shape}."
        )

    nocc = _occupation_count(mf, "mf")
    ref_nocc = _occupation_count(ref_mf, "ref_mf")
    if nocc != ref_nocc:
        raise ValueError(f"Occupation mismatch: current nocc={nocc}, ref nocc={ref_nocc}")
    return nocc


def procrustes_rotation(overlap: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
    """Return the orthogonal polar factor used for ``C_aligned = C @ Q``.

    The deterministic null-space completion is retained for basis values.  Its
    analytic derivative is deliberately not supported by
    :func:`procrustes_rotation_derivative`.
    """
    overlap = _validate_square_real("overlap", overlap)
    if rcond < 0:
        raise ValueError("rcond must be non-negative")

    u, singular_values, vt = np.linalg.svd(overlap, full_matrices=True)
    if singular_values.size == 0:
        return np.eye(overlap.shape[0])

    cutoff = rcond * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    if rank == overlap.shape[0]:
        return u @ vt

    v = vt.T
    rotation = u[:, :rank] @ v[:, :rank].T
    u_null = u[:, rank:]
    v_null = v[:, rank:]
    u_fix, _, vt_fix = np.linalg.svd(u_null.T @ v_null)
    rotation += u_null @ (u_fix @ vt_fix) @ v_null.T
    return rotation


def procrustes_rotation_derivative(
    overlap: np.ndarray,
    overlap_derivative: np.ndarray,
    rcond: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthogonal polar factor and its directional derivatives.

    ``overlap_derivative`` may have arbitrary leading perturbation dimensions
    and must end in the same two matrix dimensions as ``overlap``.
    """
    overlap = _validate_square_real("overlap", overlap)
    overlap_derivative = np.asarray(overlap_derivative)
    if overlap_derivative.shape[-2:] != overlap.shape:
        raise ValueError(
            "overlap_derivative must end in the overlap matrix shape; got "
            f"{overlap_derivative.shape} and {overlap.shape}."
        )
    if np.iscomplexobj(overlap_derivative):
        raise TypeError("overlap_derivative must be real-valued")
    if not np.all(np.isfinite(overlap_derivative)):
        raise ValueError("overlap_derivative contains non-finite values")
    if rcond < 0:
        raise ValueError("rcond must be non-negative")

    u, singular_values, vt = np.linalg.svd(overlap, full_matrices=True)
    if singular_values.size == 0:
        q = np.eye(overlap.shape[0])
        return q, np.zeros_like(overlap_derivative)

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    ratio = sigma_min / sigma_max if sigma_max else 0.0
    if ratio <= rcond:
        raise ProcrustesDerivativeError(
            "The orthogonal polar-factor derivative is not uniquely supported: "
            f"sigma_min={sigma_min:.6e}, sigma_max={sigma_max:.6e}, "
            f"condition_number={np.inf if sigma_min == 0 else sigma_max / sigma_min:.6e}, "
            f"sigma_min/sigma_max={ratio:.6e}, derivative_rcond={rcond:.6e}."
        )
    if rcond > 0 and ratio <= 10.0 * rcond:
        warnings.warn(
            "The Procrustes matching matrix is close to the derivative cutoff: "
            f"sigma_min/sigma_max={ratio:.3e}, cutoff={rcond:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    v = vt.T
    q = u @ vt
    rhs = q.T @ overlap_derivative - np.swapaxes(overlap_derivative, -1, -2) @ q
    rhs_hat = v.T @ rhs @ v
    denominator = singular_values[:, None] + singular_values[None, :]
    omega_hat = rhs_hat / denominator
    omega_hat = 0.5 * (omega_hat - np.swapaxes(omega_hat, -1, -2))
    omega = v @ omega_hat @ v.T
    dq = q @ omega
    return q, dq


def split_procrustes_basis_none(
    mol: gto.Mole,
    ref_mol: gto.Mole,
    mf: scf.hf.RHF | None = None,
    ref_mf: scf.hf.RHF | None = None,
    rcond: float = 1e-8,
    density_fit: bool = False,
    df_basis: str | None = None,
    conv_tol: float = 1e-12,
    conv_tol_grad: float = 1e-10,
    conv_tol_cpscf: float = 1e-10,
    max_cycle: int = 100,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, SplitProcrustesInfo]:
    """Build the AO-to-split-Procrustes basis at ``mol``.

    The fixed reference must have the same atoms, AO basis, AO ordering, and
    RHF occupation as the test molecule.  Coordinates may differ.
    """
    if mf is None:
        mf = run_hf(
            mol,
            density_fit=density_fit,
            df_basis=df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
    if ref_mf is None:
        ref_mf = run_hf(
            ref_mol,
            density_fit=density_fit,
            df_basis=df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
    nocc = _validate_calculations(mol, ref_mol, mf, ref_mf)

    c = np.asarray(mf.mo_coeff)
    c_ref = np.asarray(ref_mf.mo_coeff)
    c_occ, c_vrt = c[:, :nocc], c[:, nocc:]
    c_ref_occ, c_ref_vrt = c_ref[:, :nocc], c_ref[:, nocc:]

    occ_overlap = c_occ.T @ c_ref_occ
    vrt_overlap = c_vrt.T @ c_ref_vrt
    occ_rotation = procrustes_rotation(occ_overlap, rcond=rcond)
    vrt_rotation = procrustes_rotation(vrt_overlap, rcond=rcond)
    basis = np.hstack((c_occ @ occ_rotation, c_vrt @ vrt_rotation))

    s = np.asarray(mf.get_ovlp())
    orthogonality_error = basis.T @ s @ basis - np.eye(basis.shape[1])
    if np.max(np.abs(orthogonality_error), initial=0.0) > 1e-8:
        raise RuntimeError("Aligned basis is not orthonormal in the current AO metric")

    if not return_info:
        return basis
    return basis, SplitProcrustesInfo(
        nocc=nocc,
        occ_overlap=occ_overlap,
        vrt_overlap=vrt_overlap,
        occ_rotation=occ_rotation,
        vrt_rotation=vrt_rotation,
    )


def _normalise_atmlst(mol: gto.Mole, atmlst: Sequence[int] | None) -> tuple[int, ...]:
    atoms = tuple(range(mol.natm)) if atmlst is None else tuple(int(atom) for atom in atmlst)
    if len(set(atoms)) != len(atoms):
        raise ValueError("atmlst must not contain duplicate atom indices")
    if any(atom < 0 or atom >= mol.natm for atom in atoms):
        raise IndexError(f"atmlst contains an atom outside [0, {mol.natm})")
    return atoms


def ao_overlap_derivatives(
    mol: gto.Mole,
    atmlst: Sequence[int] | None = None,
) -> np.ndarray:
    """Return ``dS[A, xyz, ao, ao]`` for selected nuclear centres."""
    atoms = _normalise_atmlst(mol, atmlst)
    nao = mol.nao_nr()
    one_sided = -mol.intor("int1e_ipovlp", comp=3)
    aoslices = mol.aoslice_by_atom()
    derivatives = np.zeros((len(atoms), 3, nao, nao))
    for output_atom, atom in enumerate(atoms):
        p0, p1 = aoslices[atom, 2:]
        derivatives[output_atom, :, p0:p1] += one_sided[:, p0:p1]
        derivatives[output_atom, :, :, p0:p1] += one_sided[:, p0:p1].transpose(0, 2, 1)
    return derivatives


def rhf_mo_coefficient_derivatives(
    mf: scf.hf.RHF,
    *,
    atmlst: Sequence[int] | None = None,
    cphf_max_cycle: int = 50,
    cphf_level_shift: float = 0.0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, RHFMOResponseInfo]:
    """Return all RHF MO coefficient derivatives in a symmetric block gauge.

    PySCF supplies the occupied columns through its RHF Hessian CPHF solver.
    Calling ``mf.Hessian()`` deliberately preserves PySCF's dynamic dispatch;
    density-fitted RHF objects therefore use the DF Hessian and DF J/K response.
    The virtual columns are completed from differentiated AO-metric
    orthonormality with ``U_vv = -S_vv / 2``.
    """
    _validate_rhf(mf, "mf")
    mol = mf.mol
    atoms = _normalise_atmlst(mol, atmlst)
    c = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    mo_energy = np.asarray(mf.mo_energy)
    nocc = _occupation_count(mf, "mf")
    nao, nmo = c.shape
    if nao != nmo:
        raise ValueError("A complete square AO-to-MO coefficient matrix is required")

    hessian_factory = getattr(mf, "Hessian", None)
    hessian = (
        rhf_hessian.Hessian(mf)
        if hessian_factory is None
        else hessian_factory()
    )
    hessian.max_cycle = cphf_max_cycle
    hessian.level_shift = cphf_level_shift
    h1ao = hessian.make_h1(c, mo_occ, chkfile=None, atmlst=atoms)
    mo1_occ_by_atom, mo_e1_by_atom = hessian.solve_mo1(
        mo_energy,
        c,
        mo_occ,
        h1ao,
        atmlst=atoms,
    )
    d_c_occ = np.asarray([mo1_occ_by_atom[atom] for atom in atoms])
    mo_e1 = np.asarray([mo_e1_by_atom[atom] for atom in atoms])
    d_s = ao_overlap_derivatives(mol, atoms)
    s = np.asarray(mf.get_ovlp())

    s_mo = np.einsum("pi,axpq,qj->axij", c, d_s, c, optimize=True)
    u_occ = np.einsum("pi,pq,axqj->axij", c, s, d_c_occ, optimize=True)
    occupied_gauge_error = u_occ[:, :, :nocc] + 0.5 * s_mo[:, :, :nocc, :nocc]
    if np.max(np.abs(occupied_gauge_error), initial=0.0) > 1e-7:
        raise RuntimeError("PySCF occupied response does not match the expected overlap gauge")

    u = np.zeros((len(atoms), 3, nmo, nmo), dtype=c.dtype)
    u[:, :, :, :nocc] = u_occ
    u[:, :, :nocc, nocc:] = (
        -s_mo[:, :, :nocc, nocc:]
        - np.swapaxes(u_occ[:, :, nocc:, :], -1, -2)
    )
    u[:, :, nocc:, nocc:] = -0.5 * s_mo[:, :, nocc:, nocc:]

    d_c = np.einsum("pi,axij->axpj", c, u, optimize=True)
    d_c[:, :, :, :nocc] = d_c_occ

    orthogonality_derivative = (
        np.einsum("axpi,pq,qj->axij", d_c, s, c, optimize=True)
        + s_mo
        + np.einsum("pi,pq,axqj->axij", c, s, d_c, optimize=True)
    )
    if np.max(np.abs(orthogonality_derivative), initial=0.0) > 1e-7:
        raise RuntimeError("RHF coefficient response violates differentiated orthonormality")

    if not return_info:
        return d_c
    return d_c, RHFMOResponseInfo(
        atom_indices=atoms,
        overlap_derivative=d_s,
        mo_response=u,
        occupied_coefficient_derivative=d_c_occ,
        orbital_energy_derivative=mo_e1,
    )


def split_procrustes_basis_none_derivative(
    mol: gto.Mole,
    ref_mol: gto.Mole,
    mf: scf.hf.RHF | None = None,
    ref_mf: scf.hf.RHF | None = None,
    *,
    atmlst: Sequence[int] | None = None,
    rcond: float = 1e-8,
    derivative_rcond: float = 1e-7,
    cphf_max_cycle: int = 50,
    cphf_level_shift: float = 0.0,
    density_fit: bool = False,
    df_basis: str | None = None,
    conv_tol: float = 1e-12,
    conv_tol_grad: float = 1e-10,
    conv_tol_cpscf: float = 1e-10,
    max_cycle: int = 100,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, SplitProcrustesDerivativeInfo]:
    """Compute analytic nuclear derivatives of the split-Procrustes basis.

    At the test geometry, the canonical RHF coefficient matrix is partitioned
    into occupied and virtual blocks, ``C_occ`` and ``C_vrt``.  Each block is
    independently aligned to the corresponding fixed reference coefficients
    using the orthogonal polar factors

    ``Q_p = polar(C_p.T @ C_ref_p)``, for ``p`` in ``{occ, vrt}``.

    The resulting AO-to-split-Procrustes transformation is

    ``C_split = [C_occ @ Q_occ, C_vrt @ Q_vrt]``.

    This function returns its analytic nuclear derivative.  PySCF's RHF
    Hessian/CPHF machinery supplies the occupied orbital response, the virtual
    response is completed in the symmetric within-virtual gauge, and the two
    polar factors are differentiated directly.  The reference MO coefficients
    are held fixed: neither the reference geometry nor ``ref_mf.mo_coeff`` is
    differentiated.  The Procrustes objective uses raw Euclidean AO
    coefficient products; no same- or cross-geometry AO overlap enters it.

    Args:
        mol:
            Test-geometry PySCF ``gto.Mole`` object.  Its AO definition and atom
            ordering must match ``ref_mol``.  Molecular symmetry should be
            disabled for a smoothly differentiable geometry path.
        ref_mol:
            Fixed-reference PySCF ``gto.Mole`` object.  It may have different
            nuclear coordinates from ``mol``, but must have the same atoms, AO
            dimensions, AO labels, shell ordering, basis exponents,
            contractions, and Cartesian/spherical convention.
        mf:
            Optional converged, real-valued molecular RHF object for ``mol``.
            Conventional RHF and PySCF density-fitted RHF objects are
            supported.  For a density-fitted object, ``mf.Hessian()`` selects
            PySCF's DF Hessian and DF J/K response automatically.  If omitted,
            :func:`evcont.basis.basis_utils.run_hf` performs a conventional
            (non-density-fitted) RHF
            calculation.
        ref_mf:
            Optional converged, real-valued RHF object for ``ref_mol``.  Its MO
            coefficients define the fixed reference occupied and virtual sets.
            It must have the same number of occupied orbitals as ``mf``.  If
            omitted, :func:`evcont.basis.basis_utils.run_hf` performs a
            conventional RHF calculation.
        atmlst:
            Optional sequence of zero-based test-molecule atom indices to
            differentiate.  ``None`` selects every atom.  The leading return
            dimension follows the order supplied here and has length
            ``natom_selected = len(atmlst)`` (or ``mol.natm`` for ``None``).
            Duplicate and out-of-range indices are rejected.
        rcond:
            Non-negative relative singular-value cutoff used when constructing
            the unperturbed occupied and virtual Procrustes rotations.  A
            singular value is treated as nonzero when
            ``sigma > rcond * sigma_max``.  The default is ``1e-8``.
        derivative_rcond:
            Relative full-rank threshold for differentiating each polar factor.
            Both matching matrices must satisfy
            ``sigma_min / sigma_max > derivative_rcond``.  It must be greater
            than or equal to ``rcond``.  The default is ``1e-7``.
        cphf_max_cycle:
            Maximum number of PySCF CPHF iterations used for the occupied RHF
            response.  The default is ``50``.
        cphf_level_shift:
            Floating-point level shift passed to PySCF's CPHF solver.  The
            default is ``0.0``.
        density_fit:
            If ``True``, internally generated RHF objects use PySCF density
            fitting.  Explicit ``mf`` and ``ref_mf`` objects are used as
            supplied.
        df_basis:
            Optional auxiliary basis passed to ``density_fit`` when
            ``density_fit=True``.
        return_info:
            If ``False``, return only the derivative tensor.  If ``True``, also
            return a :class:`SplitProcrustesDerivativeInfo` containing the
            intermediate response, matching, rotation, and conditioning data.

    Returns:
        np.ndarray or tuple[np.ndarray, SplitProcrustesDerivativeInfo]:
            With ``return_info=False``, returns a real array ``d_c_split`` with
            shape ``(natom_selected, 3, nao, nmo)``.  Its element
            ``d_c_split[a, xyz, mu, p]`` is
            ``d C_split[mu, p] / d R[atom_indices[a], xyz]``, where Cartesian
            components ``xyz = 0, 1, 2`` denote ``x, y, z``.  PySCF stores
            nuclear coordinates in Bohr, so the tensor has units of AO
            coefficient per Bohr.  Molecular RHF requires the complete square
            transformation used here, so ``nmo == nao``.

            With ``return_info=True``, returns ``(d_c_split, info)``.  The
            fields of ``info`` have the following types and shapes, where
            ``nsel = natom_selected`` and ``nvir = nmo - nocc``:

            - ``atom_indices``: ``tuple[int, ...]`` of length ``nsel``.
            - ``nocc``: ``int``.
            - ``overlap_derivative``: ``(nsel, 3, nao, nao)``.
            - ``mo_coefficient_derivative``: ``(nsel, 3, nao, nmo)``.
            - ``mo_response``: ``(nsel, 3, nmo, nmo)``.
            - ``occ_overlap`` and ``occ_rotation``: ``(nocc, nocc)``.
            - ``vrt_overlap`` and ``vrt_rotation``: ``(nvir, nvir)``.
            - ``occ_singular_values``: ``(nocc,)``.
            - ``vrt_singular_values``: ``(nvir,)``.
            - ``occ_overlap_derivative`` and
              ``occ_rotation_derivative``: ``(nsel, 3, nocc, nocc)``.
            - ``vrt_overlap_derivative`` and
              ``vrt_rotation_derivative``: ``(nsel, 3, nvir, nvir)``.

    Raises:
        TypeError:
            If either mean-field object is not molecular RHF, is a DFT object,
            or contains complex-valued orbital data.
        ValueError:
            If current/reference AO definitions, coefficient dimensions, or
            RHF occupations are incompatible; if ``atmlst`` contains duplicate
            indices; or if the rank thresholds are invalid.
        IndexError:
            If ``atmlst`` contains an atom index outside ``[0, mol.natm)``.
        ProcrustesDerivativeError:
            If an occupied or virtual matching matrix is singular or too poorly
            conditioned under ``derivative_rcond``.  The exception reports its
            extreme singular values and condition number.
        RuntimeError:
            If an RHF calculation is unconverged or if the assembled response
            fails internal gauge, rotation-consistency, or differentiated
            orthonormality checks.

    Notes:
        The returned array is the derivative of the numerical AO coefficient
        matrix.  A derivative of real-space orbitals must additionally include
        derivatives of the atom-centred AO functions themselves.
    """
    if derivative_rcond < rcond:
        raise ValueError("derivative_rcond must be greater than or equal to rcond")
    if mf is None:
        mf = run_hf(
            mol,
            density_fit=density_fit,
            df_basis=df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
    if ref_mf is None:
        ref_mf = run_hf(
            ref_mol,
            density_fit=density_fit,
            df_basis=df_basis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_cpscf=conv_tol_cpscf,
            max_cycle=max_cycle,
        )
    nocc = _validate_calculations(mol, ref_mol, mf, ref_mf)

    basis, basis_info = split_procrustes_basis_none(
        mol,
        ref_mol,
        mf=mf,
        ref_mf=ref_mf,
        rcond=rcond,
        density_fit=density_fit,
        df_basis=df_basis,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_cpscf=conv_tol_cpscf,
        max_cycle=max_cycle,
        return_info=True,
    )
    d_c, response_info = rhf_mo_coefficient_derivatives(
        mf,
        atmlst=atmlst,
        cphf_max_cycle=cphf_max_cycle,
        cphf_level_shift=cphf_level_shift,
        return_info=True,
    )

    c = np.asarray(mf.mo_coeff)
    c_ref = np.asarray(ref_mf.mo_coeff)
    c_occ, c_vrt = c[:, :nocc], c[:, nocc:]
    c_ref_occ, c_ref_vrt = c_ref[:, :nocc], c_ref[:, nocc:]
    d_c_occ, d_c_vrt = d_c[:, :, :, :nocc], d_c[:, :, :, nocc:]

    d_occ_overlap = np.einsum("axpi,pj->axij", d_c_occ, c_ref_occ, optimize=True)
    d_vrt_overlap = np.einsum("axpi,pj->axij", d_c_vrt, c_ref_vrt, optimize=True)
    occ_rotation, d_occ_rotation = procrustes_rotation_derivative(
        basis_info.occ_overlap,
        d_occ_overlap,
        rcond=derivative_rcond,
    )
    vrt_rotation, d_vrt_rotation = procrustes_rotation_derivative(
        basis_info.vrt_overlap,
        d_vrt_overlap,
        rcond=derivative_rcond,
    )
    if not np.allclose(occ_rotation, basis_info.occ_rotation, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Occupied value and derivative routines selected different rotations")
    if not np.allclose(vrt_rotation, basis_info.vrt_rotation, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Virtual value and derivative routines selected different rotations")

    d_basis_occ = (
        np.einsum("axpi,ij->axpj", d_c_occ, occ_rotation, optimize=True)
        + np.einsum("pi,axij->axpj", c_occ, d_occ_rotation, optimize=True)
    )
    d_basis_vrt = (
        np.einsum("axpi,ij->axpj", d_c_vrt, vrt_rotation, optimize=True)
        + np.einsum("pi,axij->axpj", c_vrt, d_vrt_rotation, optimize=True)
    )
    d_basis = np.concatenate((d_basis_occ, d_basis_vrt), axis=-1)

    s = np.asarray(mf.get_ovlp())
    differentiated_orthogonality = (
        np.einsum("axpi,pq,qj->axij", d_basis, s, basis, optimize=True)
        + np.einsum(
            "pi,axpq,qj->axij",
            basis,
            response_info.overlap_derivative,
            basis,
            optimize=True,
        )
        + np.einsum("pi,pq,axqj->axij", basis, s, d_basis, optimize=True)
    )
    if np.max(np.abs(differentiated_orthogonality), initial=0.0) > 1e-7:
        raise RuntimeError("Split-basis derivative violates differentiated orthonormality")

    if not return_info:
        return d_basis
    return d_basis, SplitProcrustesDerivativeInfo(
        atom_indices=response_info.atom_indices,
        nocc=nocc,
        overlap_derivative=response_info.overlap_derivative,
        mo_coefficient_derivative=d_c,
        mo_response=response_info.mo_response,
        occ_overlap=basis_info.occ_overlap,
        vrt_overlap=basis_info.vrt_overlap,
        occ_rotation=occ_rotation,
        vrt_rotation=vrt_rotation,
        occ_singular_values=np.linalg.svd(basis_info.occ_overlap, compute_uv=False),
        vrt_singular_values=np.linalg.svd(basis_info.vrt_overlap, compute_uv=False),
        occ_overlap_derivative=d_occ_overlap,
        vrt_overlap_derivative=d_vrt_overlap,
        occ_rotation_derivative=d_occ_rotation,
        vrt_rotation_derivative=d_vrt_rotation,
    )
