"""Pointwise least-change occupied/virtual orbital gauges for PySCF RHF.

The construction starts from a full, atom-labelled, S-orthonormal anchor frame
``B = [B_occ, B_vir]``.  For a target S-orthonormal sector ``C``, it returns

    F = C polar(C^H S B),

where ``polar(M)`` is the unitary polar factor of ``M``.  This is the unique
least-change (orthogonal Procrustes) frame while ``C^H S B`` is nonsingular.
Only the occupied and virtual *subspaces* of the supplied RHF solution enter,
so rotations or phases of its canonical orbitals do not affect the result.

With ``use_sap=True``, the same construction is applied first from the atom
anchor to the occupied/virtual spectral projectors of ``H_core + V_SAP``, and
then from those SAP-anchored frames to the RHF subspaces.  Raw SAP eigenvector
gauges are therefore not exposed.

A fixed atom-distributed template can be applied as ``B(R) @ T``, where the
built-in atom frame ``B(R)`` is constructed and phase-fixed at the current
geometry while the unitary coordinate transform ``T`` is frozen once.  The
helpers :func:`prepare_atom_coordinate_anchor` and
:func:`prepare_frozen_mo_anchor` construct reusable keyword arguments from one
reference RHF geometry for, respectively, this co-moving atom-coordinate gauge
and a literal frozen-MO-coefficient gauge.

Primary references
------------------
P.-O. Loewdin, J. Chem. Phys. 18, 365 (1950),
https://doi.org/10.1063/1.1747632

P. H. Schoenemann, Psychometrika 31, 1 (1966),
https://doi.org/10.1007/BF02289451

N. J. Higham, SIAM J. Sci. Stat. Comput. 7, 1160 (1986),
https://doi.org/10.1137/0907079

J. Gerratt and I. M. Mills, J. Chem. Phys. 49, 1719 (1968),
https://doi.org/10.1063/1.1670299

P. Pulay, Mol. Phys. 17, 197 (1969),
https://doi.org/10.1080/00268976900100941

A. E. Reed, R. B. Weinstock, and F. Weinhold, J. Chem. Phys. 83, 735
(1985), https://doi.org/10.1063/1.449486

S. Lehtola, J. Chem. Theory Comput. 15, 1593 (2019),
https://doi.org/10.1021/acs.jctc.8b01089; and J. Chem. Phys. 152, 144105
(2020), https://doi.org/10.1063/5.0004046
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


__all__ = [
    "GaugeConditionWarning",
    "construct_least_change_orbitals",
    "prepare_atom_coordinate_anchor",
    "prepare_frozen_mo_anchor",
]


_CPHF_REFINEMENT_BATCH_SIZE = 16


class GaugeConditionWarning(RuntimeWarning):
    """Warning that a pointwise orbital gauge is becoming ill-conditioned."""


def prepare_atom_coordinate_anchor(
    mol: Any,
    mf: Any,
    *,
    anchor: str = "meta_lowdin",
    pre_orth_ao: str | np.ndarray | dict[str, Any] | None = "ANO",
    rank_tolerance: float = 1.0e-10,
    verification_tolerance: float = 1.0e-8,
    require_converged: bool = True,
) -> dict[str, Any]:
    """Prepare a fixed atom-coordinate template from one reference RHF.

    The reference occupied and virtual MOs are first ordered by their RHF
    occupations.  If ``B0`` is the phase-fixed atom-labelled frame at the
    reference geometry, this routine freezes the unitary polar factor of

    ``T_raw = B0.conj().T @ S0 @ [C_occ, C_vir]``.

    At another geometry, :func:`construct_least_change_orbitals` constructs the
    same atom-labelled frame ``B(R)`` using only that geometry and applies the
    fixed template as ``B(R) @ T``.  Thus the coefficients in atom-labelled
    coordinates and their occupied/virtual membership are global constants,
    although the physical atom-centred functions move with the nuclei.

    Parameters
    ----------
    mol, mf
        Built PySCF molecule and converged closed-shell RHF object at the one
        reference geometry.  ``mf.mol`` must be ``mol``.
    anchor
        Geometry-local atom-labelled base frame.  ``"meta_lowdin"`` is
        recommended; ``"lowdin"`` is also accepted.  Density-dependent
        molecular NAOs are deliberately rejected for this strict fixed-frame
        helper.
    pre_orth_ao
        Atomic pre-orthogonalization used by the meta-Loewdin frame.  Named or
        dictionary definitions are resolved once at the reference geometry;
        the returned keyword mapping contains the resulting fixed numerical
        AO-coordinate frame so repeated geometries do not reload and project
        the atomic reference basis.
    rank_tolerance
        Relative positive-rank threshold used for metric and polar factors.
    verification_tolerance
        Absolute tolerance for reference-frame orthogonality checks.
    require_converged
        Raise if the reference RHF has not converged.

    Returns
    -------
    dict
        Keyword arguments accepted directly by
        :func:`construct_least_change_orbitals`.  The mapping contains the base
        anchor definition, fixed ``anchor_transform``, explicit occupied
        indices, reference-MO labels, and a geometry-independent AO-basis
        fingerprint.

    Notes
    -----
    The returned NumPy transform is content-hashed in the gauge diagnostics.
    Save this returned mapping (or at least the transform and fingerprint)
    rather than recomputing it independently in different runs if bitwise
    reproducibility of the declared gauge is required.
    Phases and internal rotations of the supplied reference MO frame become
    part of the frozen convention; this helper makes that convention global
    but cannot make an exactly degenerate reference eigenspace intrinsically
    unique without an additional convention.

    The symmetric metric frame follows P.-O. Loewdin, J. Chem. Phys. 18, 365
    (1950), https://doi.org/10.1063/1.1747632.  The polar least-change factor
    follows N. J. Higham, SIAM J. Sci. Stat. Comput. 7, 1160 (1986),
    https://doi.org/10.1137/0907079.
    """
    try:
        from pyscf import lo
    except ImportError as exc:  # pragma: no cover - exercised in PySCF environments
        raise ImportError("prepare_atom_coordinate_anchor requires PySCF") from exc

    method = str(anchor).lower().replace("-", "_")
    if method not in ("meta_lowdin", "lowdin"):
        raise ValueError(
            "anchor must be 'meta_lowdin' or 'lowdin' for a fixed "
            "atom-coordinate template"
        )

    s, reference_mos, nocc, mo_labels = _prepare_reference_rhf_frame(
        mol,
        mf,
        rank_tolerance=rank_tolerance,
        verification_tolerance=verification_tolerance,
        require_converged=require_converged,
    )
    resolved_pre_orth_ao = pre_orth_ao
    if (
        method == "meta_lowdin"
        and pre_orth_ao is not None
        and not isinstance(pre_orth_ao, np.ndarray)
    ):
        resolved_pre_orth_ao = np.asarray(
            lo.orth.restore_ao_character(mol, pre_orth_ao)
        )
    built, anchor_name, _, _, built_in = _build_anchor(
        mol, mf, s, method, resolved_pre_orth_ao, lo
    )
    if not built_in or anchor_name not in ("meta_lowdin", "lowdin"):
        raise RuntimeError("The requested reference frame was not a built-in fixed atom frame")
    atom_frame, _ = _metric_orthonormalize(
        built, s, rank_tolerance, "reference atom-labelled frame"
    )
    label_overlaps = _normalized_label_overlaps(atom_frame, s)
    minimum_label_overlap = float(np.min(label_overlaps))
    if minimum_label_overlap <= rank_tolerance:
        raise np.linalg.LinAlgError(
            "The reference atom-frame phase convention is undefined: minimum same-label "
            f"overlap {minimum_label_overlap:.3e} <= {rank_tolerance:.3e}."
        )
    atom_frame = _fix_column_phases(atom_frame, s)

    raw_transform = atom_frame.conj().T @ s @ reference_mos
    transform, _ = _metric_orthonormalize(
        raw_transform,
        np.eye(raw_transform.shape[0], dtype=raw_transform.dtype),
        rank_tolerance,
        "reference atom-coordinate transform",
    )
    transform_error = float(
        np.linalg.norm(transform.conj().T @ transform - np.eye(transform.shape[0]))
    )
    if transform_error > verification_tolerance:
        raise RuntimeError(
            "The prepared atom-coordinate transform is not unitary after polar repair: "
            f"residual {transform_error:.3e}."
        )

    return {
        "anchor": method,
        "pre_orth_ao": resolved_pre_orth_ao,
        "anchor_transform": transform,
        "anchor_occ_indices": tuple(range(nocc)),
        "anchor_column_labels": mo_labels,
        "reference_basis_fingerprint": _basis_fingerprint(mol),
        "fix_anchor_phases": True,
    }


def prepare_frozen_mo_anchor(
    mol: Any,
    mf: Any,
    *,
    rank_tolerance: float = 1.0e-10,
    verification_tolerance: float = 1.0e-8,
    require_converged: bool = True,
) -> dict[str, Any]:
    """Prepare a literal frozen reference-MO coefficient anchor.

    The complete reference RHF coefficient frame is reordered as occupied then
    virtual and frozen as a numerical AO-coefficient array.  At each new
    geometry the main constructor interprets those same coefficients in the
    consistently ordered moving AO basis and symmetrically orthonormalizes the
    full frame in the current AO metric before its fixed column partition is
    aligned to the current RHF projectors.

    Parameters
    ----------
    mol, mf
        Built PySCF molecule and converged closed-shell RHF object at the one
        reference geometry.  ``mf.mol`` must be ``mol``.
    rank_tolerance
        Relative positive-rank threshold for the reference AO metric.
    verification_tolerance
        Absolute tolerance for reference-frame orthogonality checks.
    require_converged
        Raise if the reference RHF has not converged.

    Returns
    -------
    dict
        Keyword arguments accepted directly by
        :func:`construct_least_change_orbitals`, including the frozen full MO
        frame, fixed occupied indices, labels, and reference AO-basis
        fingerprint.

    Notes
    -----
    Unlike :func:`prepare_atom_coordinate_anchor`, full-frame metric
    orthonormalization of the frozen coefficients can mix the reference
    occupied and virtual columns away from the reference geometry.  This is a
    well-defined smooth full-frame gauge, but the atom-coordinate template is
    preferable when exact preservation of the frozen column partition in a
    co-moving orthonormal frame is desired.
    Reference-MO phases and rotations are intentionally frozen as supplied, so
    the returned coefficient array should be persisted rather than regenerated
    if the reference contains exactly or nearly degenerate orbital blocks.

    The symmetric metric orthonormalization follows P.-O. Loewdin,
    J. Chem. Phys. 18, 365 (1950), https://doi.org/10.1063/1.1747632; the
    subsequent least-change alignment follows P. H. Schoenemann,
    Psychometrika 31, 1 (1966), https://doi.org/10.1007/BF02289451.
    """
    _, reference_mos, nocc, mo_labels = _prepare_reference_rhf_frame(
        mol,
        mf,
        rank_tolerance=rank_tolerance,
        verification_tolerance=verification_tolerance,
        require_converged=require_converged,
    )
    return {
        "anchor": reference_mos.copy(),
        "anchor_occ_indices": tuple(range(nocc)),
        "anchor_column_labels": mo_labels,
        "reference_basis_fingerprint": _basis_fingerprint(mol),
        "fix_anchor_phases": False,
    }


def construct_least_change_orbitals(
    mol: Any,
    mf: Any,
    *,
    anchor: str | np.ndarray | Callable[[Any, Any, np.ndarray], Any] = "meta_lowdin",
    pre_orth_ao: str | np.ndarray | dict[str, Any] | None = "ANO",
    anchor_transform: np.ndarray | None = None,
    anchor_column_labels: Sequence[str] | None = None,
    reference_basis_fingerprint: str | None = None,
    anchor_occ_indices: Sequence[int] | None = None,
    anchor_selection: str = "first",
    use_sap: bool = False,
    sap_basis: str | dict[str, Any] | None = None,
    sap_hamiltonian: np.ndarray | Callable[[Any, Any], np.ndarray] | None = None,
    angle_warning_degrees: float = 70.0,
    rank_tolerance: float = 1.0e-10,
    gap_warning: float = 1.0e-3,
    selection_gap_warning: float = 1.0e-3,
    overlap_condition_warning: float = 1.0e8,
    anchor_label_warning: float = 1.0e-3,
    verification_tolerance: float = 1.0e-8,
    require_converged: bool = True,
    check_stability: bool = False,
    stability_external: bool = False,
    fix_anchor_phases: bool | None = None,
    emit_warnings: bool = True,
    return_derivatives: bool = False,
    derivative_atom_indices: Sequence[int] | None = None,
    cphf_max_cycle: int = 50,
    cphf_level_shift: float = 0.0,
    return_diagnostics: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, np.ndarray, dict[str, Any]]
):
    """Construct a pointwise least-change orbital frame from a converged RHF.

    Parameters
    ----------
    mol
        A built :class:`pyscf.gto.Mole` with a full-rank AO overlap matrix.
        Atom order is part of the gauge definition: identical atoms are
        distinguished by their PySCF atom indices.
    mf
        A closed-shell PySCF RHF object whose ``mf.mol is mol``.  Requiring the
        exact object prevents accidental reuse of orbitals from a neighboring
        geometry.  Its occupied and virtual
        canonical-orbital phases and internal rotations are discarded.
    anchor
        Full atom-labelled reference frame.  The built-in choices are
        ``"meta_lowdin"`` (the recommended fixed ANO/meta-Loewdin frame),
        ``"lowdin"`` (pure symmetric Loewdin), and ``"nao"`` (PySCF molecular
        NAOs).  NAOs depend on ``mf.make_rdm1()`` and can acquire additional
        ambiguities at degenerate natural occupations.  An ``(nao, nao)`` AO
        coefficient array or a callable ``anchor(mol, mf, S)`` is also
        accepted.  A callable may return ``B``, ``(B, occupied_indices)``, or
        ``(B, occupied_indices, column_labels)``.  Custom labels describe
        anchor columns; they are not assumed to equal AO display labels.  A
        custom builder is responsible for fixing its own internal column gauge.
    pre_orth_ao
        Fixed atomic pre-orthogonalization passed explicitly to PySCF for the
        meta-Loewdin anchor.  ``"ANO"`` is the robust PySCF default used here.
        :func:`prepare_atom_coordinate_anchor` resolves a named/dictionary
        definition once and returns the reusable numerical frame, avoiding
        repeated atomic-basis restoration in geometry scans.
    anchor_transform
        Optional fixed ``(nao, nao)`` coordinate transform applied *after* the
        base anchor has been metric-orthonormalized and its atomic phase
        convention fixed.  Its unitary polar factor is used, so a constant
        unitary ``T`` defines the co-moving frame ``B(R) @ T`` without mixing
        its fixed occupied/virtual column membership.  Use
        :func:`prepare_atom_coordinate_anchor` to construct this input from one
        reference RHF geometry.
    anchor_column_labels
        Optional labels for the final anchor columns.  These override built-in
        or callable labels and are especially useful for transformed or frozen
        reference-MO frames.
    reference_basis_fingerprint
        Optional geometry-independent basis fingerprint.  A mismatch raises
        before construction, preventing reuse of a reference anchor with a
        different atom order, AO basis, ECP/pseudopotential, spherical versus
        Cartesian convention, or AO label ordering.  Both reference helpers
        supply this value automatically.
    anchor_occ_indices
        Exactly ``N/2`` column indices defining the occupied part of the
        anchor.  Supplying this predeclared, atom-labelled set gives the
        strongest geometry-independent definition.  All remaining anchor
        columns, in their original order, define the virtual part.
    anchor_selection
        Automatic selection used only if ``anchor_occ_indices`` is absent:
        ``"first"`` (the fixed default) takes the first ``N/2`` columns;
        ``"sad"`` ranks anchor populations in a pointwise PySCF SAD density;
        and ``"hf"`` ranks them in the converged RHF density.  Population
        selection is deterministic at one geometry but can switch between
        geometries; use explicit indices for a chemically balanced fixed chart.
    use_sap
        If true, use the two-stage atom-anchor -> SAP-projector -> RHF-projector
        construction.  If false, map the anchor directly to RHF.
    sap_basis
        PySCF Gaussian SAP fit basis.  ``None`` uses ``mf.sap_basis`` (normally
        ``"sapgrasplarge"``).  It is ignored only when a custom
        ``sap_hamiltonian`` is supplied.
    sap_hamiltonian
        Optional AO-basis full SAP one-electron Hamiltonian, or callable
        ``H(mol, mf)``.  It must include ``T + V_nuc + V_SAP``.  This escape
        hatch is useful across PySCF versions and for decorated/custom core
        Hamiltonians.  Without it, the routine reproduces PySCF's
        nonrelativistic module-level ``scf.hf.get_hcore + make_sap`` operator.
        It requires ``use_sap=True``.
    angle_warning_degrees
        Warn if the largest principal angle in either occupied or virtual
        Procrustes problem exceeds this value.
    rank_tolerance
        Dimensionless failure threshold for overlap relative eigenvalues and
        principal cosines.  A rank-deficient polar problem has no unique gauge
        and raises instead of accepting an arbitrary SVD null-space completion.
    gap_warning
        Warn when the RHF or SAP occupied/virtual boundary gap, in Hartree, is
        below this value.  A nonpositive gap is rejected.
    selection_gap_warning
        Warn when automatic anchor-population selection has a smaller cutoff
        gap than this value (spatial-orbital occupation units).
    overlap_condition_warning
        Warn when the AO overlap condition number exceeds this value.
    anchor_label_warning
        Warn when the smallest normalized overlap between anchor column ``i``
        and AO label ``i`` is below this value.  A vanishing label overlap is a
        phase-chart boundary for ``fix_anchor_phases``.
    verification_tolerance
        Independent absolute tolerance for S-orthogonality, subspace leakage,
        projector, and eigensolver verification.  It is deliberately not tied
        to the scientific ``rank_tolerance`` used to reject ill-conditioned
        gauge charts.
    require_converged
        Raise on an unconverged RHF when true; otherwise construct the frame and
        emit a warning.
    check_stability
        Run PySCF's RHF stability analysis and record its status.  This can be
        appreciably more expensive than constructing the orbitals.
    stability_external
        Include RHF-to-UHF/complex external stability when
        ``check_stability=True``.  Internal RHF stability is always checked.
    fix_anchor_phases
        Fix each full-frame anchor phase by making its overlap with the
        same-index normalized AO real and nonnegative.  ``None`` enables this
        only for built-in atom-labelled frames; arbitrary arrays/callables have
        no implied one-column/one-AO semantics.  A zero phase-anchor overlap is
        rejected rather than inheriting an arbitrary upstream phase.
    emit_warnings
        Emit :class:`GaugeConditionWarning` in addition to collecting warning
        strings in the diagnostics.
    return_derivatives
        Also solve the nuclear RHF CPHF equations and return the analytic AO
        coefficient derivatives.  The derivative tensor has shape
        ``(len(derivative_atom_indices), 3, nao, nao)`` and units
        coefficient/Bohr.  Its axes are ``(atom, Cartesian, AO, orbital)``.
        These are derivatives of the numerical coefficient matrix in PySCF's
        moving atom-centred AO representation; AO-overlap (Pulay) derivatives
        are included in both orthonormality and gauge transport.
        This is currently implemented for direct RHF transport with fixed
        anchor membership and ``"lowdin"``, ``"meta_lowdin"``, or a frozen
        numerical anchor.  SAP transport, NAO anchors, callable anchors, and
        SAD/HF-population anchor selection raise :class:`NotImplementedError`
        rather than returning an incomplete derivative.
    derivative_atom_indices
        Atom indices whose three Cartesian derivatives are requested, in the
        desired output order.  ``None`` requests every atom.  This argument is
        only meaningful when ``return_derivatives=True``.
    cphf_max_cycle
        Maximum PySCF CPHF iterations for the nuclear MO response.
    cphf_level_shift
        Optional nonnegative virtual-space level shift, in Hartree, passed to
        PySCF's CPHF solver.  Zero is recommended for the most accurate
        derivative; a small shift can aid a difficult response solve.
    return_diagnostics
        Include the diagnostics mapping in the returned tuple.  When false,
        reporting-only SCF gradients, metadata fingerprints, singular-value
        tables, and diagnostic dictionaries are not evaluated; numerical
        verification and warning thresholds remain active.

    Returns
    -------
    numpy.ndarray or tuple
        AO coefficients ``coeff`` with ``coeff.conj().T @ S @ coeff = I``.
        Columns ``:N/2`` span exactly the RHF occupied space and the remaining
        columns span its S-orthogonal virtual complement.  With derivatives
        but no diagnostics the return is ``(coeff, dcoeff)``; with both it is
        ``(coeff, dcoeff, diagnostics)``.  The diagnostics contain gaps,
        principal cosines/angles, polar condition numbers, response residuals,
        atom-anchor identities, and all warning messages.

    Notes
    -----
    Except for explicitly supplied frozen reference data, this routine reads no
    orbitals, phases, or densities from another nuclear geometry.  Even with a
    fixed reference template, evaluating a new geometry uses no neighboring
    geometry.  Its result is unique and locally smooth only while every
    reported polar cross-overlap is nonsingular, the selected anchor membership
    is unchanged, and the relevant HF/SAP spectral gaps remain open.  It never
    performs neighboring-geometry phase matching or silently changes anchors.
    These conditions do not exclude a switch between distinct converged RHF
    stationary solutions; use ``check_stability=True`` and an SCF branch
    protocol appropriate to the application.

    The pointwise orbital gauge does not by itself guarantee a smooth
    many-electron eigenvector at a genuine many-body degeneracy; there one must
    transport an isolated many-body subspace rather than a single state.

    The least-change formula is the orthogonal Procrustes/polar solution of
    Schoenemann (1966) and Higham (1986).  The built-in atom frames use the
    Loewdin and NAO constructions cited in the module docstring.  PySCF's SAP
    implementation follows Lehtola (2019, 2020).  The nuclear response follows
    the perturbed-Hartree--Fock and moving-basis derivative framework of
    Gerratt and Mills (1968) and Pulay (1969), using PySCF's RHF CPHF/Hessian
    implementation for the occupied-projector response.
    """
    try:
        from pyscf import __version__ as pyscf_version
        from pyscf import lo
    except ImportError as exc:  # pragma: no cover - exercised in PySCF environments
        raise ImportError("construct_least_change_orbitals requires PySCF") from exc

    messages: list[str] = []

    def record_warning(message: str) -> None:
        if return_diagnostics:
            messages.append(message)
        if emit_warnings:
            warnings.warn(message, GaugeConditionWarning, stacklevel=3)

    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    if verification_tolerance <= 0.0:
        raise ValueError("verification_tolerance must be positive")
    if not (0.0 < angle_warning_degrees < 90.0):
        raise ValueError("angle_warning_degrees must lie strictly between 0 and 90")
    if derivative_atom_indices is not None and not return_derivatives:
        raise ValueError(
            "derivative_atom_indices was supplied but return_derivatives=False"
        )
    if not isinstance(cphf_max_cycle, (int, np.integer)) or int(cphf_max_cycle) <= 0:
        raise ValueError("cphf_max_cycle must be a positive integer")
    if not np.isfinite(cphf_level_shift) or cphf_level_shift < 0.0:
        raise ValueError("cphf_level_shift must be nonnegative")

    if return_derivatives:
        selection_mode = str(anchor_selection).lower().replace("-", "_")
        if use_sap:
            raise NotImplementedError(
                "Analytic derivatives are not yet implemented for SAP-mediated "
                "transport because the nuclear derivative of the SAP spectral "
                "projector is not exposed.  Use direct transport or finite differences."
            )
        if anchor_occ_indices is None and selection_mode in (
            "sad",
            "hf",
            "hf_population",
        ):
            raise NotImplementedError(
                "Analytic derivatives are not defined for SAD/HF population-based "
                "anchor selection: its discrete occupied membership can switch with "
                "geometry.  Supply fixed anchor_occ_indices instead."
            )
        if callable(anchor):
            raise NotImplementedError(
                "Analytic derivatives of callable anchors are not supported because no "
                "nuclear anchor-derivative callback is currently accepted."
            )
        if isinstance(anchor, str) and anchor.lower().replace("-", "_") == "nao":
            raise NotImplementedError(
                "Analytic derivatives of density-dependent molecular NAO anchors are "
                "not currently supported.  Use a lowdin/meta_lowdin or frozen anchor."
            )

    if getattr(mf, "mol", None) is not mol:
        raise ValueError(
            "mf.mol must be the same Mole object as mol.  This prevents accidentally "
            "combining orbitals from another geometry with the current AO metric."
        )

    nao = int(mol.nao_nr())
    derivative_atoms: tuple[int, ...] = ()
    if return_derivatives:
        if derivative_atom_indices is None:
            derivative_atoms = tuple(range(int(mol.natm)))
        else:
            raw_atoms = tuple(derivative_atom_indices)
            if any(not isinstance(index, (int, np.integer)) for index in raw_atoms):
                raise TypeError("derivative_atom_indices must contain only integer atom indices")
            derivative_atoms = tuple(int(index) for index in raw_atoms)
        if len(set(derivative_atoms)) != len(derivative_atoms):
            raise ValueError("derivative_atom_indices must not contain duplicates")
        if any(index < 0 or index >= int(mol.natm) for index in derivative_atoms):
            raise IndexError("derivative_atom_indices contains an out-of-range atom index")
    basis_fingerprint = (
        _basis_fingerprint(mol)
        if reference_basis_fingerprint is not None or return_diagnostics
        else None
    )
    if reference_basis_fingerprint is not None:
        reference_basis_fingerprint = str(reference_basis_fingerprint)
        if basis_fingerprint != reference_basis_fingerprint:
            raise ValueError(
                "The current atom order/AO basis fingerprint does not match the frozen "
                "reference anchor.  Reusing its coefficients or atom-coordinate template "
                "would not define the same gauge."
            )
    if int(getattr(mol, "spin", 0)) != 0 or int(mol.nelectron) % 2:
        raise ValueError("A closed-shell, even-electron molecule (mol.spin == 0) is required")
    nocc = int(mol.nelectron) // 2
    if nocc <= 0 or nocc > nao:
        raise ValueError(f"Require 0 < N/2 <= nao; got nocc={nocc} and nao={nao}")

    if require_converged and not bool(getattr(mf, "converged", False)):
        raise ValueError("The supplied RHF calculation is not converged")
    if not bool(getattr(mf, "converged", False)):
        record_warning("The supplied RHF is unconverged; its projector may not vary smoothly.")

    class_id = f"{type(mf).__module__}.{type(mf).__name__}".lower()
    if any(token in class_id for token in ("uhf", "rohf", "ghf", "dhf")):
        raise TypeError(f"Expected an RHF object, not {type(mf).__name__}")
    if hasattr(mf, "xc"):
        raise TypeError("Expected an RHF object; Kohn-Sham objects are not accepted")

    s = _hermitize(np.asarray(mf.get_ovlp(mol)))
    if s.shape != (nao, nao):
        raise ValueError(f"AO overlap has shape {s.shape}, expected {(nao, nao)}")
    s_eig = np.linalg.eigvalsh(s)
    s_min = float(s_eig[0])
    s_max = float(s_eig[-1])
    if not np.isfinite(s_eig).all() or s_max <= 0.0 or s_min / s_max <= rank_tolerance:
        raise np.linalg.LinAlgError(
            "The AO overlap is not positive definite at the requested relative rank tolerance: "
            f"lambda_min/lambda_max={s_min / s_max if s_max else np.nan:.3e}."
        )
    s_condition = s_max / s_min
    if s_condition > overlap_condition_warning:
        record_warning(
            f"AO overlap condition number is {s_condition:.3e}; the fixed AO chart is poorly conditioned."
        )

    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if mo_coeff.ndim != 2 or mo_occ.ndim != 1:
        raise TypeError("Expected spatial RHF mo_coeff and mo_occ arrays")
    if mo_coeff.shape != (nao, nao) or mo_occ.shape != (nao,):
        raise ValueError(
            "A full square RHF orbital frame is required; got "
            f"mo_coeff {mo_coeff.shape} and mo_occ {mo_occ.shape}. "
            "Check whether PySCF pruned linearly dependent AOs."
        )
    occ_mask = np.isclose(mo_occ, 2.0, atol=1.0e-7, rtol=0.0)
    vir_mask = np.isclose(mo_occ, 0.0, atol=1.0e-7, rtol=0.0)
    if not np.all(occ_mask | vir_mask) or int(np.count_nonzero(occ_mask)) != nocc:
        raise ValueError("RHF occupations must contain exactly N/2 values of 2 and only values 2 or 0")

    c_occ_raw = mo_coeff[:, occ_mask]
    c_vir_raw = mo_coeff[:, vir_mask]
    c_input = np.hstack((c_occ_raw, c_vir_raw))
    input_mo_orth_error = float(np.linalg.norm(c_input.conj().T @ s @ c_input - np.eye(nao)))
    input_cross_error = float(np.linalg.norm(c_occ_raw.conj().T @ s @ c_vir_raw))
    if input_cross_error > verification_tolerance:
        raise ValueError(
            f"RHF occupied and virtual orbitals are not S-orthogonal (residual {input_cross_error:.3e})"
        )
    if input_mo_orth_error > verification_tolerance:
        record_warning(
            "The input RHF frame has S-orthogonality error "
            f"{input_mo_orth_error:.3e}; each sector will be symmetrically repaired."
        )
    c_occ, _ = _metric_orthonormalize(c_occ_raw, s, rank_tolerance, "RHF occupied block")
    c_vir, _ = _metric_orthonormalize(c_vir_raw, s, rank_tolerance, "RHF virtual block")

    mo_energy = np.asarray(getattr(mf, "mo_energy", np.empty(0)))
    if mo_energy.shape != (nao,):
        raise ValueError("A full RHF mo_energy array is required for gap diagnostics")
    hf_gap: float | None
    if c_vir.shape[1]:
        hf_gap = float(np.min(np.real(mo_energy[vir_mask])) - np.max(np.real(mo_energy[occ_mask])))
        hf_spectral_width = float(np.ptp(np.real(mo_energy)))
        hf_gap_floor = rank_tolerance * max(1.0, hf_spectral_width)
        if hf_gap <= hf_gap_floor:
            raise ValueError(
                "The RHF occupied/virtual energy gap is numerically closed: "
                f"{hf_gap:.3e} Eh <= {hf_gap_floor:.3e} Eh"
            )
        if hf_gap < gap_warning:
            record_warning(
                f"The RHF occupied/virtual gap is only {hf_gap:.3e} Eh; projector smoothness is fragile."
            )
    else:
        hf_gap = None

    effective_pre_orth_ao = pre_orth_ao
    if (
        return_derivatives
        and isinstance(anchor, str)
        and anchor.lower().replace("-", "_") == "meta_lowdin"
        and pre_orth_ao is not None
        and not isinstance(pre_orth_ao, np.ndarray)
    ):
        # The atom-local restoration coefficients are fixed in the moving AO
        # representation.  Resolve them once because the analytic derivative
        # path and the value path both need the same frame.
        effective_pre_orth_ao = np.asarray(
            lo.orth.restore_ao_character(mol, pre_orth_ao)
        )
    built_anchor, anchor_name, callable_occ_indices, supplied_anchor_labels, built_in_anchor = _build_anchor(
        mol, mf, s, anchor, effective_pre_orth_ao, lo
    )
    if built_anchor.shape != (nao, nao):
        raise ValueError(f"Anchor has shape {built_anchor.shape}, expected a full {(nao, nao)} frame")
    raw_anchor_gram = _hermitize(built_anchor.conj().T @ s @ built_anchor)
    raw_anchor_orth_error = float(np.linalg.norm(raw_anchor_gram - np.eye(nao)))
    if raw_anchor_orth_error > verification_tolerance:
        anchor_description = (
            "frozen reference coefficient frame"
            if reference_basis_fingerprint is not None and isinstance(anchor, np.ndarray)
            else "supplied base anchor"
        )
        record_warning(
            f"The {anchor_description} has S-orthogonality error "
            f"{raw_anchor_orth_error:.3e}; the full frame will be symmetrically orthonormalized."
        )
    anchor_frame, anchor_gram_eig = _metric_orthonormalize(
        built_anchor, s, rank_tolerance, "full base anchor"
    )
    metric_anchor_frame = anchor_frame.copy()
    ao_labels = tuple(str(label) for label in mol.ao_labels())
    if supplied_anchor_labels is None:
        base_anchor_labels = ao_labels if built_in_anchor else tuple(
            f"anchor_column_{index}" for index in range(nao)
        )
    else:
        base_anchor_labels = tuple(str(label) for label in supplied_anchor_labels)
        if len(base_anchor_labels) != nao:
            raise ValueError(
                f"The anchor callable supplied {len(base_anchor_labels)} labels for {nao} columns"
            )

    phase_fix_enabled = built_in_anchor if fix_anchor_phases is None else bool(fix_anchor_phases)
    anchor_label_overlaps: np.ndarray | None
    if built_in_anchor or phase_fix_enabled:
        anchor_label_overlaps = _normalized_label_overlaps(anchor_frame, s)
        min_anchor_label_overlap: float | None = float(np.min(anchor_label_overlaps))
    else:
        anchor_label_overlaps = None
        min_anchor_label_overlap = None
    if phase_fix_enabled:
        assert min_anchor_label_overlap is not None
        if min_anchor_label_overlap <= rank_tolerance:
            raise np.linalg.LinAlgError(
                "The same-index AO/anchor phase convention is undefined: minimum label overlap "
                f"{min_anchor_label_overlap:.3e} <= {rank_tolerance:.3e}."
            )
        anchor_frame = _fix_column_phases(anchor_frame, s)
    if min_anchor_label_overlap is not None and min_anchor_label_overlap < anchor_label_warning:
        record_warning(
            "The minimum same-label AO/anchor overlap is "
            f"{min_anchor_label_overlap:.3e}; the anchor phase convention is near a chart boundary."
        )
    if anchor_name == "nao":
        record_warning(
            "The molecular NAO anchor depends on the RHF density and can retain arbitrary unitary "
            "rotations inside degenerate natural-occupation blocks.  Same-column phase fixing does "
            "not remove that ambiguity; use meta_lowdin or a fixed custom anchor for a strict gauge."
        )

    transform_descriptor = (
        _input_descriptor(anchor_transform) if return_diagnostics else "not_collected"
    )
    applied_anchor_transform: np.ndarray | None = None
    transform_info: dict[str, Any] | None = None
    if anchor_transform is None:
        if return_diagnostics:
            transform_info = {
                "enabled": False,
                "definition": transform_descriptor,
                "raw_unitarity_error": None,
                "gram_minimum_eigenvalue": None,
                "gram_maximum_eigenvalue": None,
                "condition_number": None,
                "repaired_by_polar_factor": False,
            }
    else:
        raw_transform = np.asarray(anchor_transform)
        if raw_transform.shape != (nao, nao):
            raise ValueError(
                f"anchor_transform has shape {raw_transform.shape}, expected {(nao, nao)}"
            )
        if not np.isfinite(raw_transform).all():
            raise ValueError("anchor_transform contains nonfinite values")
        transform_gram = _hermitize(raw_transform.conj().T @ raw_transform)
        transform_unitarity_error = float(np.linalg.norm(transform_gram - np.eye(nao)))
        fixed_transform, transform_gram_eig = _metric_orthonormalize(
            raw_transform,
            np.eye(nao, dtype=raw_transform.dtype),
            rank_tolerance,
            "fixed anchor coordinate transform",
        )
        applied_anchor_transform = fixed_transform
        if transform_unitarity_error > verification_tolerance:
            record_warning(
                "The supplied anchor_transform has unitarity error "
                f"{transform_unitarity_error:.3e}; its unique unitary polar factor will be used."
            )
        anchor_frame = anchor_frame @ fixed_transform
        if return_diagnostics:
            transform_info = {
                "enabled": True,
                "definition": transform_descriptor,
                "raw_unitarity_error": transform_unitarity_error,
                "gram_minimum_eigenvalue": float(np.min(transform_gram_eig)),
                "gram_maximum_eigenvalue": float(np.max(transform_gram_eig)),
                "condition_number": float(
                    np.sqrt(np.max(transform_gram_eig) / np.min(transform_gram_eig))
                ),
                "repaired_by_polar_factor": bool(
                    transform_unitarity_error > verification_tolerance
                ),
            }

    if anchor_column_labels is None:
        anchor_labels = (
            base_anchor_labels
            if anchor_transform is None
            else tuple(f"template_column_{index}" for index in range(nao))
        )
    else:
        anchor_labels = tuple(str(label) for label in anchor_column_labels)
        if len(anchor_labels) != nao:
            raise ValueError(
                f"anchor_column_labels contains {len(anchor_labels)} labels, expected {nao}"
            )

    final_anchor_orth_error = float(
        np.linalg.norm(anchor_frame.conj().T @ s @ anchor_frame - np.eye(nao))
    )
    if final_anchor_orth_error > verification_tolerance:
        raise RuntimeError(
            "The final transformed anchor is not S-orthonormal: residual "
            f"{final_anchor_orth_error:.3e}."
        )

    if callable_occ_indices is not None:
        if anchor_occ_indices is not None:
            raise ValueError("Occupied anchor indices were supplied both by the callable and the argument")
        anchor_occ_indices = callable_occ_indices

    selected_occ, selected_vir, selection_info = _choose_anchor_partition(
        mol=mol,
        mf=mf,
        s=s,
        anchor_frame=anchor_frame,
        nocc=nocc,
        explicit_indices=anchor_occ_indices,
        selection=anchor_selection,
    )
    selection_gap = selection_info.get("cutoff_gap")
    if selection_info["mode"] in ("sad", "hf_population"):
        record_warning(
            f"anchor_selection='{selection_info['mode']}' uses a hard pointwise population ranking. "
            "Membership can switch between sampled geometries even when both endpoint cutoff gaps "
            "look healthy; explicit anchor_occ_indices are required for a fixed global chart."
        )
    if selection_gap is not None and selection_gap < selection_gap_warning:
        record_warning(
            "The automatic occupied-anchor population cutoff gap is "
            f"{selection_gap:.3e}; anchor membership may switch under a small geometry change. "
            "Use explicit anchor_occ_indices for a fixed chart."
        )

    anchor_order = tuple(selected_occ) + tuple(selected_vir)
    b_ordered = anchor_frame[:, anchor_order]
    b_occ = b_ordered[:, :nocc]
    b_vir = b_ordered[:, nocc:]

    direct_diagnostic: dict[str, Any] | None = None
    if use_sap and return_diagnostics:
        # Only the two-stage route reports the unused direct path for
        # comparison.  The direct route obtains the same singular-value
        # diagnostics from the SVDs that actually construct its polar maps.
        direct_diagnostic = _partition_cross_diagnostics(
            c_occ, c_vir, b_occ, b_vir, s
        )
    sap_info: dict[str, Any] | None = (
        {"enabled": bool(use_sap)} if return_diagnostics else None
    )

    if sap_hamiltonian is not None and not use_sap:
        raise ValueError("sap_hamiltonian was supplied but use_sap=False")
    if sap_basis is not None and not use_sap:
        raise ValueError("sap_basis was supplied but use_sap=False")

    if use_sap:
        h_sap, sap_source = _obtain_sap_hamiltonian(
            mol, mf, sap_basis=sap_basis, sap_hamiltonian=sap_hamiltonian
        )
        h_sap = _hermitize(np.asarray(h_sap))
        if h_sap.shape != (nao, nao):
            raise ValueError(f"SAP Hamiltonian has shape {h_sap.shape}, expected {(nao, nao)}")
        h_anchor = _hermitize(b_ordered.conj().T @ h_sap @ b_ordered)
        sap_energy, sap_u = np.linalg.eigh(h_anchor)
        sap_coordinate_residual = h_anchor @ sap_u - sap_u * sap_energy[np.newaxis, :]
        c_sap = b_ordered @ sap_u
        c_sap_occ = c_sap[:, :nocc]
        c_sap_vir = c_sap[:, nocc:]

        sap_gap: float | None
        if c_sap_vir.shape[1]:
            sap_gap = float(sap_energy[nocc] - sap_energy[nocc - 1])
            sap_boundary_residual = float(
                max(
                    np.linalg.norm(sap_coordinate_residual[:, nocc - 1]),
                    np.linalg.norm(sap_coordinate_residual[:, nocc]),
                )
            )
            sap_spectral_width = float(np.ptp(sap_energy))
            sap_gap_floor = max(
                rank_tolerance * max(1.0, sap_spectral_width),
                100.0 * sap_boundary_residual,
            )
            if sap_gap <= sap_gap_floor:
                raise np.linalg.LinAlgError(
                    f"The SAP rank-{nocc} spectral projector is numerically nonunique "
                    f"(gap {sap_gap:.3e} Eh <= {sap_gap_floor:.3e} Eh)"
                )
            if sap_gap < gap_warning:
                record_warning(
                    f"The SAP occupied/virtual spectral gap is only {sap_gap:.3e} Eh; "
                    "the intermediate projector is fragile."
                )
        else:
            sap_gap = None
            sap_boundary_residual = None

        a_occ, a_vir, atom_to_sap = _align_partition(
            c_sap_occ,
            c_sap_vir,
            b_occ,
            b_vir,
            s,
            stage="atom anchor -> SAP",
            rank_tolerance=rank_tolerance,
            angle_warning_degrees=angle_warning_degrees,
            record_warning=record_warning,
            collect_diagnostics=return_diagnostics,
        )
        f_occ, f_vir, sap_to_hf = _align_partition(
            c_occ,
            c_vir,
            a_occ,
            a_vir,
            s,
            stage="SAP -> RHF",
            rank_tolerance=rank_tolerance,
            angle_warning_degrees=angle_warning_degrees,
            record_warning=record_warning,
            collect_diagnostics=return_diagnostics,
        )
        sap_residual = _generalized_eigen_residual(h_sap, s, c_sap, sap_energy)
        if sap_residual > verification_tolerance:
            record_warning(
                f"The SAP generalized-eigenproblem residual is {sap_residual:.3e}; "
                "the intermediate projector may be numerically inaccurate."
        )
        if return_diagnostics:
            assert sap_info is not None
            sap_info.update(
                {
                    "source": sap_source,
                    "gap_hartree": sap_gap,
                    "boundary_eigenpair_residual_hartree": sap_boundary_residual,
                    "eigen_residual": sap_residual,
                    "atom_to_sap": atom_to_sap,
                    "sap_to_hf": sap_to_hf,
                }
            )
            transport = {
                "route": "via_sap",
                "anchor_to_hf_direct_diagnostic": direct_diagnostic,
                "atom_to_sap": atom_to_sap,
                "sap_to_hf": sap_to_hf,
            }
        else:
            transport = None
        transport_route = "via_sap"
    else:
        f_occ, f_vir, anchor_to_hf = _align_partition(
            c_occ,
            c_vir,
            b_occ,
            b_vir,
            s,
            stage="atom anchor -> RHF",
            rank_tolerance=rank_tolerance,
            angle_warning_degrees=angle_warning_degrees,
            record_warning=record_warning,
            collect_diagnostics=return_diagnostics,
        )
        transport = (
            {"route": "direct", "anchor_to_hf": anchor_to_hf}
            if return_diagnostics
            else None
        )
        transport_route = "direct"

    coeff = np.hstack((f_occ, f_vir))
    coeff = np.real_if_close(coeff, tol=1000)
    output_orthogonality_error = float(np.linalg.norm(coeff.conj().T @ s @ coeff - np.eye(nao)))
    occupied_virtual_cross_error = float(np.linalg.norm(f_occ.conj().T @ s @ f_vir))
    occupied_leakage = float(np.linalg.norm(c_vir.conj().T @ s @ f_occ))
    virtual_leakage = float(np.linalg.norm(c_occ.conj().T @ s @ f_vir))
    occupied_projector_error = _projector_frobenius_error(c_occ, f_occ, s)
    virtual_projector_error = _projector_frobenius_error(c_vir, f_vir, s)
    verification_error = max(
        output_orthogonality_error,
        occupied_virtual_cross_error,
        occupied_leakage,
        virtual_leakage,
        occupied_projector_error,
        virtual_projector_error,
    )
    if verification_error > verification_tolerance:
        raise RuntimeError(
            "Least-change orbital verification failed; largest residual is "
            f"{verification_error:.3e} (tolerance {verification_tolerance:.3e})"
        )

    dcoeff: np.ndarray | None = None
    derivative_info: dict[str, Any] | None = None
    if return_diagnostics:
        derivative_info = {
            "requested": bool(return_derivatives),
            "atom_indices": derivative_atoms,
            "shape": None,
            "units": "coefficient/Bohr",
        }
    if return_derivatives:
        dcoeff, response_info = _construct_nuclear_derivatives(
            mol=mol,
            mf=mf,
            s=s,
            built_anchor=built_anchor,
            metric_anchor_frame=metric_anchor_frame,
            final_anchor_frame=anchor_frame,
            anchor=anchor,
            pre_orth_ao=effective_pre_orth_ao,
            lo_module=lo,
            phase_fix_enabled=phase_fix_enabled,
            anchor_transform=applied_anchor_transform,
            anchor_order=anchor_order,
            c_occ_raw=c_occ_raw,
            c_occ=c_occ,
            f_occ=f_occ,
            f_vir=f_vir,
            coeff=coeff,
            atom_indices=derivative_atoms,
            rank_tolerance=rank_tolerance,
            verification_tolerance=verification_tolerance,
            cphf_max_cycle=int(cphf_max_cycle),
            cphf_level_shift=float(cphf_level_shift),
            collect_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            assert derivative_info is not None
            derivative_info.update(response_info)
            derivative_info["shape"] = tuple(int(size) for size in dcoeff.shape)
        cphf_residual = float(response_info["cphf_residual"])
        cphf_warning_threshold = max(
            verification_tolerance,
            10.0 * float(getattr(mf, "conv_tol_cpscf", 1.0e-9)),
        )
        if cphf_residual > cphf_warning_threshold:
            record_warning(
                "The independent canonical-MO CPHF residual is "
                f"{cphf_residual:.3e}, above {cphf_warning_threshold:.3e}. "
                "The projector/orthonormality tangents remain internally consistent, "
                "but finite-difference agreement may be limited by SCF stationarity or "
                "CPHF convergence."
            )

    scf_gradient_norm: float | None = None
    if return_diagnostics:
        try:
            scf_gradient_norm = float(
                np.linalg.norm(np.asarray(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
            )
        except Exception:  # decorated RHF classes need not expose get_grad
            scf_gradient_norm = None

    stability_info: dict[str, Any] | None = None
    if return_diagnostics:
        stability_info = {
            "checked": bool(check_stability),
            "internal_stable": None,
            "external_checked": bool(check_stability and stability_external),
            "external_stable": None,
        }
    if check_stability:
        try:
            _, _, stable_internal, stable_external = mf.stability(
                internal=True,
                external=stability_external,
                return_status=True,
            )
        except Exception as exc:
            raise RuntimeError("PySCF RHF stability analysis failed") from exc
        if return_diagnostics:
            assert stability_info is not None
            stability_info["internal_stable"] = bool(stable_internal)
            stability_info["external_stable"] = (
                None if stable_external is None else bool(stable_external)
            )
        if not bool(stable_internal):
            record_warning(
                "The RHF solution is internally unstable; independent geometries may select "
                "different SCF stationary branches even when the orbital gap is open."
            )
        if stability_external and not bool(stable_external):
            record_warning(
                "The RHF solution is externally unstable to UHF/complex variations; the "
                "closed-shell branch may not represent a smooth physical ground state."
            )

    if not return_diagnostics:
        if return_derivatives:
            assert dcoeff is not None
            return coeff, dcoeff
        return coeff

    assert basis_fingerprint is not None
    assert transform_info is not None
    assert stability_info is not None
    assert transport is not None
    assert derivative_info is not None
    anchor_descriptor = _input_descriptor(anchor)
    pre_orth_descriptor = _input_descriptor(pre_orth_ao)
    sap_descriptor = (
        "disabled"
        if not use_sap
        else _input_descriptor(sap_hamiltonian)
        if sap_hamiltonian is not None
        else _input_descriptor(
            getattr(mf, "sap_basis", "sapgrasplarge") if sap_basis is None else sap_basis
        )
    )
    assert sap_info is not None
    sap_info["definition"] = sap_descriptor
    nao_aoshell_fingerprint = _definition_fingerprint(repr(getattr(lo.nao, "AOSHELL", None)))
    project_ecp_basis = getattr(lo.orth, "PROJECT_ECP_BASIS", None)
    gauge_fingerprint = _definition_fingerprint(
        (
            basis_fingerprint,
            str(pyscf_version),
            nao_aoshell_fingerprint,
            project_ecp_basis,
            anchor_name,
            anchor_descriptor,
            pre_orth_descriptor,
            transform_descriptor,
            tuple(anchor_labels),
            reference_basis_fingerprint,
            tuple(selected_occ),
            selection_info["mode"],
            phase_fix_enabled,
            transport_route,
            sap_descriptor,
        )
    )
    raw_ao_labels = mol.ao_labels(fmt=False)
    ao_keys = tuple(
        (index,) + tuple(str(part) for part in label)
        for index, label in enumerate(raw_ao_labels)
    )
    diagnostics: dict[str, Any] = {
        "method": "least_change_projected_polar",
        "pyscf_version": str(pyscf_version),
        "n_ao": nao,
        "n_occ": nocc,
        "n_vir": nao - nocc,
        "scf_converged": bool(getattr(mf, "converged", False)),
        "scf_gradient_norm": scf_gradient_norm,
        "stability": stability_info,
        "hf_gap_hartree": hf_gap,
        "overlap": {
            "minimum_eigenvalue": s_min,
            "maximum_eigenvalue": s_max,
            "relative_minimum_eigenvalue": s_min / s_max,
            "condition_number": s_condition,
        },
        "anchor": {
            "kind": anchor_name,
            "definition": anchor_descriptor,
            "pre_orth_ao": pre_orth_descriptor,
            "reference_basis_fingerprint": reference_basis_fingerprint,
            "nao_aoshell_fingerprint": nao_aoshell_fingerprint,
            "project_ecp_basis": project_ecp_basis,
            "density_dependent": anchor_name == "nao",
            "transform": transform_info,
            "selection": selection_info,
            "occupied_indices": tuple(int(i) for i in selected_occ),
            "virtual_indices": tuple(int(i) for i in selected_vir),
            "base_column_labels": base_anchor_labels,
            "column_labels": anchor_labels,
            "occupied_labels": tuple(anchor_labels[i] for i in selected_occ),
            "raw_orthogonality_error": raw_anchor_orth_error,
            "final_orthogonality_error": final_anchor_orth_error,
            "gram_minimum_eigenvalue": float(np.min(anchor_gram_eig)),
            "minimum_label_overlap": min_anchor_label_overlap,
            "label_overlaps": None
            if anchor_label_overlaps is None
            else tuple(float(x) for x in anchor_label_overlaps),
            "phase_fixed": phase_fix_enabled,
            "phase_fix_scope": "base_anchor_before_transform",
            "built_in_atom_label_semantics": bool(
                built_in_anchor and anchor_transform is None
            ),
            "base_built_in_atom_label_semantics": built_in_anchor,
        },
        "transport": transport,
        "sap": sap_info,
        "nuclear_derivatives": derivative_info,
        "verification": {
            "input_mo_orthogonality_error": input_mo_orth_error,
            "input_occ_vir_cross_error": input_cross_error,
            "output_orthogonality_error": output_orthogonality_error,
            "output_occ_vir_cross_error": occupied_virtual_cross_error,
            "occupied_leakage": occupied_leakage,
            "virtual_leakage": virtual_leakage,
            "occupied_projector_error": occupied_projector_error,
            "virtual_projector_error": virtual_projector_error,
        },
        "basis_fingerprint": basis_fingerprint,
        "gauge_definition_fingerprint": gauge_fingerprint,
        "gauge_fingerprint_scope": (
            "built_in_with_content_hashed_transform"
            if built_in_anchor and anchor_transform is not None
            else "built_in"
            if built_in_anchor
            else "custom_content"
            if isinstance(anchor, np.ndarray)
            else "custom_identity_only"
        ),
        "ao_labels": ao_labels,
        "ao_keys": ao_keys,
        "warnings": tuple(messages),
    }

    if return_derivatives:
        assert dcoeff is not None
        if return_diagnostics:
            return coeff, dcoeff, diagnostics
        return coeff, dcoeff
    if return_diagnostics:
        return coeff, diagnostics
    return coeff


def _prepare_reference_rhf_frame(
    mol: Any,
    mf: Any,
    *,
    rank_tolerance: float,
    verification_tolerance: float,
    require_converged: bool,
) -> tuple[np.ndarray, np.ndarray, int, tuple[str, ...]]:
    """Validate and order a complete reference RHF frame for public helpers."""
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    if verification_tolerance <= 0.0:
        raise ValueError("verification_tolerance must be positive")
    if getattr(mf, "mol", None) is not mol:
        raise ValueError("mf.mol must be the same Mole object as the reference mol")
    if require_converged and not bool(getattr(mf, "converged", False)):
        raise ValueError("The reference RHF calculation is not converged")
    if not bool(getattr(mf, "converged", False)):
        warnings.warn(
            "The frozen anchor is being prepared from an unconverged RHF frame.",
            GaugeConditionWarning,
            stacklevel=3,
        )

    class_id = f"{type(mf).__module__}.{type(mf).__name__}".lower()
    if any(token in class_id for token in ("uhf", "rohf", "ghf", "dhf")):
        raise TypeError(f"Expected an RHF object, not {type(mf).__name__}")
    if hasattr(mf, "xc"):
        raise TypeError("Expected an RHF object; Kohn-Sham objects are not accepted")

    nao = int(mol.nao_nr())
    if int(getattr(mol, "spin", 0)) != 0 or int(mol.nelectron) % 2:
        raise ValueError("A closed-shell, even-electron reference molecule is required")
    nocc = int(mol.nelectron) // 2
    if nocc <= 0 or nocc > nao:
        raise ValueError(f"Require 0 < N/2 <= nao; got nocc={nocc} and nao={nao}")

    s = _hermitize(np.asarray(mf.get_ovlp(mol)))
    if s.shape != (nao, nao):
        raise ValueError(f"Reference AO overlap has shape {s.shape}, expected {(nao, nao)}")
    s_eig = np.linalg.eigvalsh(s)
    if (
        not np.isfinite(s_eig).all()
        or float(s_eig[-1]) <= 0.0
        or float(s_eig[0]) / float(s_eig[-1]) <= rank_tolerance
    ):
        raise np.linalg.LinAlgError(
            "The reference AO overlap is not positive definite at the requested rank tolerance"
        )

    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if mo_coeff.shape != (nao, nao) or mo_occ.shape != (nao,):
        raise ValueError(
            "A full square reference RHF frame is required; got "
            f"mo_coeff {mo_coeff.shape} and mo_occ {mo_occ.shape}."
        )
    if not np.isfinite(mo_coeff).all() or not np.isfinite(mo_occ).all():
        raise ValueError("The reference RHF coefficients or occupations contain nonfinite values")
    occ_mask = np.isclose(mo_occ, 2.0, atol=1.0e-7, rtol=0.0)
    vir_mask = np.isclose(mo_occ, 0.0, atol=1.0e-7, rtol=0.0)
    if not np.all(occ_mask | vir_mask) or int(np.count_nonzero(occ_mask)) != nocc:
        raise ValueError("Reference RHF occupations must contain N/2 values of 2 and otherwise 0")

    c_occ_raw = mo_coeff[:, occ_mask]
    c_vir_raw = mo_coeff[:, vir_mask]
    cross_error = float(np.linalg.norm(c_occ_raw.conj().T @ s @ c_vir_raw))
    if cross_error > verification_tolerance:
        raise ValueError(
            "Reference occupied and virtual orbitals are not S-orthogonal: residual "
            f"{cross_error:.3e}."
        )
    c_occ, _ = _metric_orthonormalize(
        c_occ_raw, s, rank_tolerance, "reference RHF occupied block"
    )
    c_vir, _ = _metric_orthonormalize(
        c_vir_raw, s, rank_tolerance, "reference RHF virtual block"
    )
    reference_mos = np.hstack((c_occ, c_vir))
    orthogonality_error = float(
        np.linalg.norm(reference_mos.conj().T @ s @ reference_mos - np.eye(nao))
    )
    if orthogonality_error > verification_tolerance:
        raise ValueError(
            "The ordered reference RHF frame is not S-orthonormal after block repair: "
            f"residual {orthogonality_error:.3e}."
        )

    mo_energy = np.asarray(getattr(mf, "mo_energy", np.empty(0)))
    if mo_energy.shape != (nao,):
        raise ValueError("A full reference RHF mo_energy array is required")
    if c_vir.shape[1]:
        gap = float(np.min(np.real(mo_energy[vir_mask])) - np.max(np.real(mo_energy[occ_mask])))
        width = float(np.ptp(np.real(mo_energy)))
        gap_floor = rank_tolerance * max(1.0, width)
        if gap <= gap_floor:
            raise ValueError(
                "The reference RHF occupied/virtual gap is numerically closed: "
                f"{gap:.3e} Eh <= {gap_floor:.3e} Eh."
            )

    occupied_indices = tuple(int(i) for i in np.flatnonzero(occ_mask))
    virtual_indices = tuple(int(i) for i in np.flatnonzero(vir_mask))
    labels = tuple(f"reference occupied MO {i}" for i in occupied_indices) + tuple(
        f"reference virtual MO {i}" for i in virtual_indices
    )
    return s, reference_mos, nocc, labels


def _hermitize(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) * 0.5


def _metric_orthonormalize(
    c: np.ndarray,
    s: np.ndarray,
    rank_tolerance: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if c.shape[1] == 0:
        return c.copy(), np.empty(0)
    gram = _hermitize(c.conj().T @ s @ c)
    eig, vec = np.linalg.eigh(gram)
    largest = float(eig[-1])
    if largest <= 0.0 or float(eig[0]) / largest <= rank_tolerance:
        raise np.linalg.LinAlgError(
            f"{label} is rank deficient: lambda_min/lambda_max={float(eig[0]) / largest:.3e}"
        )
    inv_sqrt = (vec * (1.0 / np.sqrt(eig))) @ vec.conj().T
    return c @ inv_sqrt, eig


def _inverse_sqrt_with_frechet_derivatives(
    gram: np.ndarray,
    dgram: np.ndarray,
    rank_tolerance: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``G**(-1/2)`` and its Hermitian Frechet derivatives."""
    gram = _hermitize(np.asarray(gram))
    dgram = np.asarray(dgram)
    eig, vec = np.linalg.eigh(gram)
    largest = float(eig[-1]) if eig.size else 1.0
    if eig.size and (
        largest <= 0.0 or float(eig[0]) / largest <= rank_tolerance
    ):
        raise np.linalg.LinAlgError(
            f"{label} is rank deficient: "
            f"lambda_min/lambda_max={float(eig[0]) / largest:.3e}"
        )

    root = np.sqrt(eig)
    inv_sqrt = (vec * (1.0 / root)) @ vec.conj().T
    # Divided differences of f(x)=x^(-1/2).  This expression is continuous
    # when eigenvalues coincide, unlike an eigenvector-derivative formula.
    divided_difference = -1.0 / (
        root[:, None] * root[None, :] * (root[:, None] + root[None, :])
    )
    derivatives = np.empty(
        (dgram.shape[0],) + gram.shape,
        dtype=np.result_type(gram.dtype, dgram.dtype),
    )
    for perturbation, value in enumerate(dgram):
        value_eigenbasis = vec.conj().T @ _hermitize(value) @ vec
        derivatives[perturbation] = vec @ (
            divided_difference * value_eigenbasis
        ) @ vec.conj().T
    return inv_sqrt, derivatives, eig


def _metric_orthonormalize_with_derivatives(
    c: np.ndarray,
    dc: np.ndarray,
    s: np.ndarray,
    ds: np.ndarray,
    rank_tolerance: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate ``C (C^H S C)^(-1/2)`` for many perturbations."""
    c = np.asarray(c)
    dc = np.asarray(dc)
    ds = np.asarray(ds)
    if c.shape[1] == 0:
        return c.copy(), dc.copy(), np.empty(0)
    gram = _hermitize(c.conj().T @ s @ c)
    dgram = np.empty(
        (dc.shape[0], c.shape[1], c.shape[1]),
        dtype=np.result_type(c.dtype, dc.dtype, s.dtype, ds.dtype),
    )
    for perturbation in range(dc.shape[0]):
        dgram[perturbation] = _hermitize(
            dc[perturbation].conj().T @ s @ c
            + c.conj().T @ ds[perturbation] @ c
            + c.conj().T @ s @ dc[perturbation]
        )
    inverse_sqrt, d_inverse_sqrt, eig = _inverse_sqrt_with_frechet_derivatives(
        gram, dgram, rank_tolerance, label
    )
    frame = c @ inverse_sqrt
    dframe = np.empty(
        (dc.shape[0],) + frame.shape,
        dtype=np.result_type(frame.dtype, dc.dtype, d_inverse_sqrt.dtype),
    )
    for perturbation in range(dc.shape[0]):
        dframe[perturbation] = (
            dc[perturbation] @ inverse_sqrt
            + c @ d_inverse_sqrt[perturbation]
        )
    return frame, dframe, eig


def _phase_fix_with_derivatives(
    frame: np.ndarray,
    dframe: np.ndarray,
    s: np.ndarray,
    ds: np.ndarray,
    rank_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the same-index AO/anchor phase convention."""
    overlap = np.diag(s @ frame)
    ao_norm = np.sqrt(np.maximum(np.real(np.diag(s)), np.finfo(float).tiny))
    if np.any(np.abs(overlap) / ao_norm <= rank_tolerance):
        raise np.linalg.LinAlgError(
            "The anchor phase derivative is undefined at a zero same-index overlap"
        )
    phase = np.conjugate(overlap) / np.abs(overlap)
    fixed = frame * phase[np.newaxis, :]
    dtype = np.result_type(frame.dtype, dframe.dtype, s.dtype, ds.dtype, 1j)
    dfixed = np.empty(dframe.shape, dtype=dtype)
    for perturbation in range(dframe.shape[0]):
        doverlap = np.diag(
            ds[perturbation] @ frame + s @ dframe[perturbation]
        )
        dphase = -1j * phase * np.imag(doverlap / overlap)
        dfixed[perturbation] = (
            dframe[perturbation] * phase[np.newaxis, :]
            + frame * dphase[np.newaxis, :]
        )
    return np.real_if_close(fixed, tol=1000), np.real_if_close(dfixed, tol=1000)


def _nuclear_overlap_derivatives(
    mol: Any, atom_indices: Sequence[int], dtype: np.dtype[Any]
) -> np.ndarray:
    """Build total AO-overlap derivatives in PySCF's moving-AO convention."""
    nao = int(mol.nao_nr())
    result = np.zeros(
        (len(atom_indices), 3, nao, nao),
        dtype=np.result_type(dtype, float),
    )
    if not atom_indices:
        return result
    one_sided = -np.asarray(mol.intor("int1e_ipovlp", comp=3))
    aoslices = mol.aoslice_by_atom()
    for output_index, atom_index in enumerate(atom_indices):
        p0, p1 = (int(value) for value in aoslices[atom_index, 2:])
        result[output_index, :, p0:p1, :] += one_sided[:, p0:p1, :]
        result[output_index, :, :, p0:p1] += one_sided[
            :, p0:p1, :
        ].conj().transpose(0, 2, 1)
    return result


def _project_out_with_derivatives(
    raw: np.ndarray,
    draw: np.ndarray,
    basis: np.ndarray,
    dbasis: np.ndarray,
    s: np.ndarray,
    ds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate ``(I-B B^H S) raw`` for an S-orthonormal ``B``."""
    if basis.shape[1] == 0:
        return raw.copy(), draw.copy()
    overlap = basis.conj().T @ s @ raw
    projected = raw - basis @ overlap
    dprojected = np.empty(
        draw.shape,
        dtype=np.result_type(raw.dtype, draw.dtype, basis.dtype, dbasis.dtype, ds.dtype),
    )
    for perturbation in range(draw.shape[0]):
        doverlap = (
            dbasis[perturbation].conj().T @ s @ raw
            + basis.conj().T @ ds[perturbation] @ raw
            + basis.conj().T @ s @ draw[perturbation]
        )
        dprojected[perturbation] = (
            draw[perturbation]
            - dbasis[perturbation] @ overlap
            - basis @ doverlap
        )
    return projected, dprojected


def _raw_anchor_with_nuclear_derivatives(
    *,
    mol: Any,
    anchor: str | np.ndarray,
    pre_orth_ao: Any,
    lo_module: Any,
    s: np.ndarray,
    ds: np.ndarray,
    expected_anchor: np.ndarray,
    rank_tolerance: float,
    verification_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct and differentiate a supported raw anchor definition."""
    nperturbation = ds.shape[0]
    nao = int(mol.nao_nr())
    if not isinstance(anchor, str):
        value = np.asarray(anchor)
        derivative = np.zeros(
            (nperturbation,) + value.shape,
            dtype=np.result_type(value.dtype, ds.dtype),
        )
        method = "frozen_array"
    else:
        method = anchor.lower().replace("-", "_")
        if method == "lowdin":
            seed = np.eye(nao, dtype=s.dtype)
            dseed = np.zeros(
                (nperturbation, nao, nao), dtype=np.result_type(seed.dtype, ds.dtype)
            )
            value, derivative, _ = _metric_orthonormalize_with_derivatives(
                seed, dseed, s, ds, rank_tolerance, "Loewdin anchor"
            )
        elif method == "meta_lowdin":
            if pre_orth_ao is None:
                pre = np.eye(nao, dtype=s.dtype)
            elif isinstance(pre_orth_ao, np.ndarray):
                pre = np.asarray(pre_orth_ao)
            else:
                pre = np.asarray(
                    lo_module.orth.restore_ao_character(mol, pre_orth_ao)
                )
            if pre.shape != (nao, nao):
                raise ValueError(
                    f"The meta-Loewdin pre-orthogonal frame has shape {pre.shape}, "
                    f"expected {(nao, nao)}"
                )
            dpre = np.zeros(
                (nperturbation, nao, nao), dtype=np.result_type(pre.dtype, ds.dtype)
            )
            core, valence, rydberg = lo_module.nao._core_val_ryd_list(mol)
            core = tuple(int(index) for index in core)
            valence = tuple(int(index) for index in valence)
            rydberg = tuple(int(index) for index in rydberg)
            if sorted(core + valence + rydberg) != list(range(nao)):
                raise RuntimeError(
                    "PySCF's meta-Loewdin core/valence/Rydberg partition does not "
                    "cover every AO exactly once"
                )
            value = np.empty((nao, nao), dtype=np.result_type(pre.dtype, s.dtype))
            derivative = np.empty(
                (nperturbation, nao, nao),
                dtype=np.result_type(pre.dtype, s.dtype, ds.dtype),
            )

            if core:
                core_raw = pre[:, core]
                core_draw = dpre[:, :, core]
                core_frame, dcore_frame, _ = _metric_orthonormalize_with_derivatives(
                    core_raw,
                    core_draw,
                    s,
                    ds,
                    rank_tolerance,
                    "meta-Loewdin core anchor",
                )
                value[:, core] = core_frame
                derivative[:, :, core] = dcore_frame
            else:
                core_frame = np.empty((nao, 0), dtype=value.dtype)
                dcore_frame = np.empty((nperturbation, nao, 0), dtype=derivative.dtype)

            if valence:
                valence_raw, dvalence_raw = _project_out_with_derivatives(
                    pre[:, valence],
                    dpre[:, :, valence],
                    core_frame,
                    dcore_frame,
                    s,
                    ds,
                )
                valence_frame, dvalence_frame, _ = (
                    _metric_orthonormalize_with_derivatives(
                        valence_raw,
                        dvalence_raw,
                        s,
                        ds,
                        rank_tolerance,
                        "meta-Loewdin valence anchor",
                    )
                )
                value[:, valence] = valence_frame
                derivative[:, :, valence] = dvalence_frame

            core_valence = core + valence
            if rydberg:
                cv_frame = value[:, core_valence]
                dcv_frame = derivative[:, :, core_valence]
                rydberg_raw, drydberg_raw = _project_out_with_derivatives(
                    pre[:, rydberg],
                    dpre[:, :, rydberg],
                    cv_frame,
                    dcv_frame,
                    s,
                    ds,
                )
                rydberg_frame, drydberg_frame, _ = (
                    _metric_orthonormalize_with_derivatives(
                        rydberg_raw,
                        drydberg_raw,
                        s,
                        ds,
                        rank_tolerance,
                        "meta-Loewdin Rydberg anchor",
                    )
                )
                value[:, rydberg] = rydberg_frame
                derivative[:, :, rydberg] = drydberg_frame

        else:  # guarded by the public function, retained for direct helper use
            raise NotImplementedError(
                f"Analytic nuclear derivatives are not implemented for anchor={anchor!r}"
            )

        # PySCF orth_ao applies this real diagonal sign convention after both
        # Loewdin variants.  It is locally constant away from a zero diagonal.
        for column in range(nao):
            if np.real(value[column, column]) < 0.0:
                value[:, column] *= -1.0
                derivative[:, :, column] *= -1.0

    reconstruction_error = float(np.linalg.norm(value - expected_anchor))
    reconstruction_scale = max(float(np.linalg.norm(expected_anchor)), 1.0)
    if reconstruction_error / reconstruction_scale > max(
        10.0 * verification_tolerance, 1.0e-9
    ):
        raise RuntimeError(
            "Analytic anchor reconstruction does not reproduce the value used by "
            f"the orbital constructor (relative error {reconstruction_error / reconstruction_scale:.3e})."
        )
    return value, derivative, {
        "anchor_derivative_method": method,
        "raw_anchor_reconstruction_error": reconstruction_error,
    }


def _construct_nuclear_derivatives(
    *,
    mol: Any,
    mf: Any,
    s: np.ndarray,
    built_anchor: np.ndarray,
    metric_anchor_frame: np.ndarray,
    final_anchor_frame: np.ndarray,
    anchor: str | np.ndarray,
    pre_orth_ao: Any,
    lo_module: Any,
    phase_fix_enabled: bool,
    anchor_transform: np.ndarray | None,
    anchor_order: Sequence[int],
    c_occ_raw: np.ndarray,
    c_occ: np.ndarray,
    f_occ: np.ndarray,
    f_vir: np.ndarray,
    coeff: np.ndarray,
    atom_indices: Sequence[int],
    rank_tolerance: float,
    verification_tolerance: float,
    cphf_max_cycle: int,
    cphf_level_shift: float,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine PySCF nuclear CPHF with analytic anchor and polar tangents."""
    nao = int(mol.nao_nr())
    nocc = c_occ.shape[1]
    if not atom_indices:
        empty_info: dict[str, Any] = {"cphf_residual": 0.0}
        if collect_diagnostics:
            empty_info.update(
                {
                    "method": "PySCF nuclear CPHF plus analytic polar Frechet derivative",
                    "density_fitted": bool(
                        getattr(mf, "with_df", None) is not None
                    ),
                    "hessian_class": None,
                    "cphf_max_cycle": cphf_max_cycle,
                    "cphf_level_shift_hartree": cphf_level_shift,
                    "initial_cphf_residual": 0.0,
                    "cphf_refinement_applied": False,
                    "cphf_refinement_target": None,
                    "cphf_refinement_batch_size": 0,
                    "cphf_refinement_minimum_batch_size": 0,
                    "cphf_refinement_passes": 0,
                    "cphf_refinement_batches": 0,
                    "cphf_refinement_active_right_hand_sides": 0,
                    "maximum_orthonormality_tangent_residual": 0.0,
                    "maximum_subspace_tangent_residual": 0.0,
                    "maximum_projector_idempotency_tangent_residual": 0.0,
                }
            )
        return np.zeros((0, 3, nao, nao), dtype=coeff.dtype), empty_info

    ds_by_atom = _nuclear_overlap_derivatives(mol, atom_indices, s.dtype)
    ds = ds_by_atom.reshape(-1, nao, nao)
    nperturbation = ds.shape[0]

    try:
        # Importing the Hessian package registers .Hessian() for both ordinary
        # RHF and DFRHF objects; dispatch then chooses the appropriate PySCF
        # response implementation.
        from pyscf import hessian as _pyscf_hessian  # noqa: F401
        from pyscf import lib as pyscf_lib
        from pyscf.hessian import rhf as rhf_hessian

        hessian_object = mf.Hessian()
        hessian_object.max_cycle = cphf_max_cycle
        hessian_object.level_shift = cphf_level_shift
        h1ao = hessian_object.make_h1(
            mf.mo_coeff, mf.mo_occ, atmlst=list(atom_indices)
        )
        mo1_by_atom, mo_e1_by_atom = hessian_object.solve_mo1(
            mf.mo_energy,
            mf.mo_coeff,
            mf.mo_occ,
            h1ao,
            atmlst=list(atom_indices),
        )
    except Exception as exc:
        raise RuntimeError(
            "PySCF failed while solving the nuclear RHF CPHF equations for "
            "least-change orbital derivatives"
        ) from exc

    dc_occ_raw = np.asarray(
        [np.asarray(mo1_by_atom[index]) for index in atom_indices]
    ).reshape(nperturbation, nao, nocc)
    if (
        dc_occ_raw.shape != (nperturbation, nao, nocc)
        or not np.isfinite(dc_occ_raw).all()
    ):
        raise RuntimeError("PySCF returned an invalid occupied-orbital CPHF response")
    if any(mo_e1_by_atom[index] is None for index in atom_indices):
        raise RuntimeError("PySCF did not return the expected nuclear CPHF energy response")

    # Verify the response independently in the canonical MO representation.
    # PySCF's standard multi-right-hand-side Krylov solve uses a conservative
    # linear-dependence cutoff.  If that cutoff limits the residual, solve
    # normalized correction equations in small blocks with the same PySCF
    # response kernel and Krylov implementation.
    canonical_coeff = np.asarray(mf.mo_coeff)
    canonical_occ_mask = np.asarray(mf.mo_occ) > 0
    canonical_vir_mask = np.asarray(mf.mo_occ) == 0
    canonical_occ = canonical_coeff[:, canonical_occ_mask]
    epsilon = np.asarray(mf.mo_energy)
    epsilon_occ = epsilon[canonical_occ_mask]
    epsilon_vir = epsilon[canonical_vir_mask]
    energy_denominator = epsilon_vir[:, np.newaxis] - epsilon_occ[np.newaxis, :]
    cphf_u = np.asarray(
        [canonical_coeff.conj().T @ s @ value for value in dc_occ_raw]
    )
    s1_mo = np.asarray(
        [canonical_coeff.conj().T @ value @ canonical_occ for value in ds]
    )
    h1_mo = np.asarray(
        [
            canonical_coeff.conj().T
            @ np.asarray(h1ao[atom])[cartesian]
            @ canonical_occ
            for atom in atom_indices
            for cartesian in range(3)
        ]
    )
    response_kernel = rhf_hessian.gen_vind(mf, mf.mo_coeff, mf.mo_occ)

    def canonical_cphf_residual(
        response: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        induced = np.asarray(response_kernel(response))
        hs = h1_mo - s1_mo * epsilon_occ[np.newaxis, np.newaxis, :] + induced
        virtual_residual = (
            energy_denominator[np.newaxis, :, :]
            * response[:, canonical_vir_mask, :]
            + hs[:, canonical_vir_mask, :]
        )
        occupied_residual = (
            response[:, canonical_occ_mask, :]
            + 0.5 * s1_mo[:, canonical_occ_mask, :]
        )
        scale = max(
            float(np.linalg.norm(hs[:, canonical_vir_mask, :])),
            float(
                np.linalg.norm(
                    energy_denominator[np.newaxis, :, :]
                    * response[:, canonical_vir_mask, :]
                )
            ),
            1.0,
        )
        normalized = max(
            float(np.linalg.norm(virtual_residual)) / scale,
            float(np.linalg.norm(occupied_residual)),
        )
        return normalized, virtual_residual, occupied_residual

    initial_cphf_residual, virtual_residual, _ = canonical_cphf_residual(cphf_u)
    refinement_target = max(1.0e-12, min(1.0e-10, 0.1 * verification_tolerance))
    cphf_refinement_applied = initial_cphf_residual > refinement_target
    refinement_batch_size = min(_CPHF_REFINEMENT_BATCH_SIZE, nperturbation)
    refinement_minimum_batch_size = 0
    refinement_passes = 0
    refinement_batches = 0
    refinement_active_right_hand_sides = 0
    if cphf_refinement_applied:
        nmo = canonical_coeff.shape[1]
        nvir = int(np.count_nonzero(canonical_vir_mask))
        correction_tolerance = min(1.0e-13, 0.1 * refinement_target)
        correction_lindep = min(1.0e-24, 0.01 * correction_tolerance**2)

        def apply_cphf_correction(
            residual: np.ndarray, batch_size: int
        ) -> tuple[int, int]:
            number_batches = 0
            number_active_right_hand_sides = 0
            for start in range(0, nperturbation, batch_size):
                stop = min(start + batch_size, nperturbation)
                right_hand_sides = (
                    -residual[start:stop]
                    / energy_denominator[np.newaxis, :, :]
                ).reshape(stop - start, -1)
                right_hand_side_norms = np.linalg.norm(right_hand_sides, axis=1)
                active = right_hand_side_norms > np.finfo(float).tiny
                if not np.any(active):
                    continue

                normalized_right_hand_sides = (
                    right_hand_sides[active]
                    / right_hand_side_norms[active, np.newaxis]
                )
                number_active = int(np.count_nonzero(active))

                def correction_operator(vectors: np.ndarray) -> np.ndarray:
                    vectors = np.asarray(vectors)
                    input_was_vector = vectors.ndim == 1
                    vectors = vectors.reshape(-1, nvir, nocc)
                    correction_mo = np.zeros(
                        (vectors.shape[0], nmo, nocc),
                        dtype=np.result_type(cphf_u.dtype, vectors.dtype),
                    )
                    correction_mo[:, canonical_vir_mask, :] = vectors
                    induced_correction = np.asarray(response_kernel(correction_mo))
                    result = (
                        induced_correction[:, canonical_vir_mask, :]
                        / energy_denominator[np.newaxis, :, :]
                    ).reshape(vectors.shape[0], -1)
                    return result[0] if input_was_vector else result

                try:
                    normalized_correction = np.asarray(
                        pyscf_lib.krylov(
                            correction_operator,
                            normalized_right_hand_sides,
                            tol=correction_tolerance,
                            max_cycle=cphf_max_cycle,
                            lindep=correction_lindep,
                            verbose=getattr(mf, "verbose", 0),
                        )
                    ).reshape(number_active, -1)
                except Exception as exc:
                    raise RuntimeError(
                        "PySCF block CPHF residual refinement failed for nuclear "
                        f"perturbations {start}:{stop}"
                    ) from exc
                if (
                    normalized_correction.shape != (number_active, nvir * nocc)
                    or not np.isfinite(normalized_correction).all()
                ):
                    raise RuntimeError(
                        "PySCF returned an invalid block CPHF residual correction"
                    )
                correction = (
                    normalized_correction
                    * right_hand_side_norms[active, np.newaxis]
                ).reshape(number_active, nvir, nocc)
                active_rows = np.flatnonzero(active)
                for correction_index, local_index in enumerate(active_rows):
                    cphf_u[
                        start + int(local_index), canonical_vir_mask, :
                    ] += correction[correction_index]
                number_batches += 1
                number_active_right_hand_sides += number_active
            return number_batches, number_active_right_hand_sides

        # PySCF's Krylov implementation can solve several right-hand sides at
        # once, but its linear-dependence screening acts on their absolute
        # norms.  Normalize each residual before the block solve, then restore
        # its scale.  Small fixed-size batches retain most of the response/J-K
        # reuse without making the temporary density stack grow with natm.  If
        # symmetry makes a block nearly linearly dependent, halve the batch and
        # refine the remaining residual; the final batch size of one recovers
        # the robust independent-solve limit without penalizing typical cases.
        remaining_virtual_residual = virtual_residual
        current_batch_size = refinement_batch_size
        while True:
            pass_batches, pass_active = apply_cphf_correction(
                remaining_virtual_residual, current_batch_size
            )
            refinement_passes += 1
            refinement_batches += pass_batches
            refinement_active_right_hand_sides += pass_active
            refinement_minimum_batch_size = current_batch_size
            cphf_residual, remaining_virtual_residual, _ = (
                canonical_cphf_residual(cphf_u)
            )
            if (
                cphf_residual <= refinement_target
                or current_batch_size == 1
                or pass_active == 0
            ):
                break
            current_batch_size = max(1, current_batch_size // 2)
        dc_occ_raw = np.einsum(
            "pq,xqi->xpi", canonical_coeff, cphf_u, optimize=True
        )
    else:
        cphf_residual = initial_cphf_residual

    response_occ, dresponse_occ, _ = _metric_orthonormalize_with_derivatives(
        c_occ_raw,
        dc_occ_raw,
        s,
        ds,
        rank_tolerance,
        "CPHF occupied frame",
    )
    occupied_frame_reconstruction_error = float(np.linalg.norm(response_occ - c_occ))
    if occupied_frame_reconstruction_error > 10.0 * verification_tolerance:
        raise RuntimeError(
            "The CPHF occupied-frame convention cannot be reconciled with the "
            f"constructor (residual {occupied_frame_reconstruction_error:.3e})"
        )

    density_projector_kernel = response_occ @ response_occ.conj().T
    occupied_projector = density_projector_kernel @ s
    d_occupied_projector = np.empty(
        (nperturbation, nao, nao),
        dtype=np.result_type(response_occ.dtype, dresponse_occ.dtype, ds.dtype),
    )
    for perturbation in range(nperturbation):
        dkernel = (
            dresponse_occ[perturbation] @ response_occ.conj().T
            + response_occ @ dresponse_occ[perturbation].conj().T
        )
        d_occupied_projector[perturbation] = (
            dkernel @ s + density_projector_kernel @ ds[perturbation]
        )

    raw_anchor, draw_anchor, anchor_info = _raw_anchor_with_nuclear_derivatives(
        mol=mol,
        anchor=anchor,
        pre_orth_ao=pre_orth_ao,
        lo_module=lo_module,
        s=s,
        ds=ds,
        expected_anchor=built_anchor,
        rank_tolerance=rank_tolerance,
        verification_tolerance=verification_tolerance,
    )
    response_anchor, dresponse_anchor, anchor_gram_eig = (
        _metric_orthonormalize_with_derivatives(
            raw_anchor,
            draw_anchor,
            s,
            ds,
            rank_tolerance,
            "full base anchor derivative",
        )
    )
    metric_anchor_reconstruction_error = float(
        np.linalg.norm(response_anchor - metric_anchor_frame)
    )
    if metric_anchor_reconstruction_error > 10.0 * verification_tolerance:
        raise RuntimeError(
            "The analytic metric-anchor reconstruction differs from the constructor: "
            f"residual {metric_anchor_reconstruction_error:.3e}"
        )

    if phase_fix_enabled:
        response_anchor, dresponse_anchor = _phase_fix_with_derivatives(
            response_anchor,
            dresponse_anchor,
            s,
            ds,
            rank_tolerance,
        )
    if anchor_transform is not None:
        response_anchor = response_anchor @ anchor_transform
        dresponse_anchor = np.einsum(
            "xmp,pq->xmq", dresponse_anchor, anchor_transform, optimize=True
        )
    final_anchor_reconstruction_error = float(
        np.linalg.norm(response_anchor - final_anchor_frame)
    )
    if final_anchor_reconstruction_error > 10.0 * verification_tolerance:
        raise RuntimeError(
            "The analytic final-anchor reconstruction differs from the constructor: "
            f"residual {final_anchor_reconstruction_error:.3e}"
        )

    order = tuple(int(index) for index in anchor_order)
    ordered_anchor = response_anchor[:, order]
    dordered_anchor = dresponse_anchor[:, :, order]
    anchor_occ = ordered_anchor[:, :nocc]
    anchor_vir = ordered_anchor[:, nocc:]
    danchor_occ = dordered_anchor[:, :, :nocc]
    danchor_vir = dordered_anchor[:, :, nocc:]

    identity = np.eye(nao, dtype=occupied_projector.dtype)
    virtual_projector = identity - occupied_projector
    d_virtual_projector = -d_occupied_projector

    def differentiate_projected_polar(
        projector: np.ndarray,
        dprojector: np.ndarray,
        block_anchor: np.ndarray,
        dblock_anchor: np.ndarray,
        expected: np.ndarray,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        projected = projector @ block_anchor
        dprojected = np.empty(
            dblock_anchor.shape,
            dtype=np.result_type(
                projector.dtype, dprojector.dtype, block_anchor.dtype, dblock_anchor.dtype
            ),
        )
        for perturbation in range(nperturbation):
            dprojected[perturbation] = (
                dprojector[perturbation] @ block_anchor
                + projector @ dblock_anchor[perturbation]
            )
        aligned, daligned, gram_eig = _metric_orthonormalize_with_derivatives(
            projected,
            dprojected,
            s,
            ds,
            rank_tolerance,
            label,
        )
        reconstruction_error = float(np.linalg.norm(aligned - expected))
        if reconstruction_error > 10.0 * verification_tolerance:
            raise RuntimeError(
                f"The analytic {label} value differs from the SVD constructor: "
                f"residual {reconstruction_error:.3e}"
            )
        return aligned, daligned, gram_eig, reconstruction_error

    response_f_occ, df_occ, occupied_polar_eig, occupied_reconstruction_error = (
        differentiate_projected_polar(
            occupied_projector,
            d_occupied_projector,
            anchor_occ,
            danchor_occ,
            f_occ,
            "occupied projected-polar map",
        )
    )
    response_f_vir, df_vir, virtual_polar_eig, virtual_reconstruction_error = (
        differentiate_projected_polar(
            virtual_projector,
            d_virtual_projector,
            anchor_vir,
            danchor_vir,
            f_vir,
            "virtual projected-polar map",
        )
    )
    response_coeff = np.hstack((response_f_occ, response_f_vir))
    dcoeff = np.concatenate((df_occ, df_vir), axis=2)
    dcoeff = np.real_if_close(dcoeff, tol=1000).reshape(
        len(atom_indices), 3, nao, nao
    )
    coefficient_reconstruction_error = float(np.linalg.norm(response_coeff - coeff))

    flat_dcoeff = dcoeff.reshape(nperturbation, nao, nao)
    # Verify one perturbation at a time.  Materializing all four residual
    # stacks would require about three additional full dC-sized arrays.
    maximum_orthonormality_tangent = 0.0
    maximum_occupied_subspace_tangent = 0.0
    maximum_virtual_subspace_tangent = 0.0
    maximum_projector_tangent = 0.0
    virtual_space_projector = identity - occupied_projector
    for perturbation, value in enumerate(flat_dcoeff):
        orthonormality_tangent = (
            value.conj().T @ s @ coeff
            + coeff.conj().T @ ds[perturbation] @ coeff
            + coeff.conj().T @ s @ value
        )
        maximum_orthonormality_tangent = max(
            maximum_orthonormality_tangent,
            float(np.linalg.norm(orthonormality_tangent)),
        )

        occupied_subspace_tangent = (
            virtual_space_projector @ df_occ[perturbation]
            - d_occupied_projector[perturbation] @ f_occ
        )
        maximum_occupied_subspace_tangent = max(
            maximum_occupied_subspace_tangent,
            float(np.linalg.norm(occupied_subspace_tangent)),
        )

        virtual_subspace_tangent = (
            occupied_projector @ df_vir[perturbation]
            + d_occupied_projector[perturbation] @ f_vir
        )
        maximum_virtual_subspace_tangent = max(
            maximum_virtual_subspace_tangent,
            float(np.linalg.norm(virtual_subspace_tangent)),
        )

        projector_tangent = (
            d_occupied_projector[perturbation] @ occupied_projector
            + occupied_projector @ d_occupied_projector[perturbation]
            - d_occupied_projector[perturbation]
        )
        maximum_projector_tangent = max(
            maximum_projector_tangent,
            float(np.linalg.norm(projector_tangent)),
        )
    maximum_subspace_tangent = max(
        maximum_occupied_subspace_tangent,
        maximum_virtual_subspace_tangent,
    )
    tangent_verification_error = max(
        maximum_orthonormality_tangent,
        maximum_subspace_tangent,
        maximum_projector_tangent,
    )
    if tangent_verification_error > verification_tolerance:
        raise RuntimeError(
            "Least-change nuclear derivative verification failed; largest response "
            f"residual is {tangent_verification_error:.3e}"
        )

    if not collect_diagnostics:
        return dcoeff, {"cphf_residual": cphf_residual}

    derivative_norms = tuple(
        tuple(float(np.linalg.norm(dcoeff[atom, cartesian])) for cartesian in range(3))
        for atom in range(len(atom_indices))
    )
    if len(atom_indices) == int(mol.natm) and set(atom_indices) == set(range(int(mol.natm))):
        translation_sum_norm: float | None = float(
            max(
                np.linalg.norm(np.sum(dcoeff[:, cartesian], axis=0))
                for cartesian in range(3)
            )
        )
    else:
        translation_sum_norm = None

    info = {
        "method": "PySCF nuclear CPHF plus analytic polar Frechet derivative",
        "density_fitted": bool(getattr(mf, "with_df", None) is not None),
        "hessian_class": f"{type(hessian_object).__module__}.{type(hessian_object).__name__}",
        "cphf_max_cycle": cphf_max_cycle,
        "cphf_level_shift_hartree": cphf_level_shift,
        "cphf_convergence_tolerance": float(getattr(mf, "conv_tol_cpscf", np.nan)),
        "initial_cphf_residual": initial_cphf_residual,
        "cphf_refinement_applied": cphf_refinement_applied,
        "cphf_refinement_target": refinement_target,
        "cphf_refinement_batch_size": refinement_batch_size,
        "cphf_refinement_minimum_batch_size": refinement_minimum_batch_size,
        "cphf_refinement_passes": refinement_passes,
        "cphf_refinement_batches": refinement_batches,
        "cphf_refinement_active_right_hand_sides": (
            refinement_active_right_hand_sides
        ),
        "cphf_residual": cphf_residual,
        "maximum_orthonormality_tangent_residual": maximum_orthonormality_tangent,
        "maximum_subspace_tangent_residual": maximum_subspace_tangent,
        "maximum_projector_idempotency_tangent_residual": maximum_projector_tangent,
        "occupied_frame_reconstruction_error": occupied_frame_reconstruction_error,
        "metric_anchor_reconstruction_error": metric_anchor_reconstruction_error,
        "final_anchor_reconstruction_error": final_anchor_reconstruction_error,
        "coefficient_reconstruction_error": coefficient_reconstruction_error,
        "occupied_polar_reconstruction_error": occupied_reconstruction_error,
        "virtual_polar_reconstruction_error": virtual_reconstruction_error,
        "anchor_gram_minimum_eigenvalue": float(np.min(anchor_gram_eig)),
        "occupied_minimum_principal_cosine": float(np.sqrt(np.min(occupied_polar_eig))),
        "virtual_minimum_principal_cosine": float(np.sqrt(np.min(virtual_polar_eig)))
        if virtual_polar_eig.size
        else 1.0,
        "maximum_derivative_frobenius_norm": float(
            max(np.linalg.norm(value) for value in flat_dcoeff)
        ),
        "derivative_frobenius_norms_by_atom_xyz": derivative_norms,
        "rigid_translation_sum_norm": translation_sum_norm,
    }
    info.update(anchor_info)
    return dcoeff, info


def _build_anchor(
    mol: Any,
    mf: Any,
    s: np.ndarray,
    anchor: str | np.ndarray | Callable[[Any, Any, np.ndarray], Any],
    pre_orth_ao: Any,
    lo_module: Any,
) -> tuple[np.ndarray, str, Sequence[int] | None, Sequence[str] | None, bool]:
    supplied_indices: Sequence[int] | None = None
    supplied_labels: Sequence[str] | None = None
    built_in = False
    if isinstance(anchor, str):
        method = anchor.lower().replace("-", "_")
        if method == "meta_lowdin":
            value = lo_module.orth.orth_ao(
                mol, method="meta_lowdin", pre_orth_ao=pre_orth_ao, s=s
            )
        elif method == "lowdin":
            value = lo_module.orth.orth_ao(
                mol, method="lowdin", pre_orth_ao=None, s=s
            )
        elif method == "nao":
            value = lo_module.orth.orth_ao(mf, method="nao", s=s)
        else:
            raise ValueError("anchor must be 'meta_lowdin', 'lowdin', 'nao', an array, or a callable")
        name = method
        built_in = True
    elif callable(anchor):
        value = anchor(mol, mf, s)
        module = getattr(anchor, "__module__", type(anchor).__module__)
        qualname = getattr(anchor, "__qualname__", getattr(anchor, "__name__", type(anchor).__name__))
        name = f"{module}.{qualname}"
        if isinstance(value, tuple):
            if len(value) == 2:
                value, supplied_indices = value
            elif len(value) == 3:
                value, supplied_indices, supplied_labels = value
            else:
                raise ValueError(
                    "An anchor callable tuple return must be (full_frame, occupied_indices) "
                    "or (full_frame, occupied_indices, column_labels)"
                )
    else:
        value = anchor
        name = "array"
    return np.asarray(value), name, supplied_indices, supplied_labels, built_in


def _normalized_label_overlaps(frame: np.ndarray, s: np.ndarray) -> np.ndarray:
    if frame.shape[1] == 0:
        return np.empty(0)
    overlaps = np.diag(s @ frame)
    ao_norms = np.sqrt(np.maximum(np.real(np.diag(s)), np.finfo(float).tiny))
    return np.abs(overlaps) / ao_norms


def _fix_column_phases(frame: np.ndarray, s: np.ndarray) -> np.ndarray:
    dtype = np.result_type(frame.dtype, s.dtype)
    fixed = np.array(frame, dtype=dtype, copy=True)
    diagonal_overlaps = np.diag(s @ fixed)
    for index, overlap in enumerate(diagonal_overlaps):
        magnitude = abs(overlap)
        if magnitude > 0.0:
            fixed[:, index] *= np.conjugate(overlap) / magnitude
    return fixed


def _choose_anchor_partition(
    *,
    mol: Any,
    mf: Any,
    s: np.ndarray,
    anchor_frame: np.ndarray,
    nocc: int,
    explicit_indices: Sequence[int] | None,
    selection: str,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    nao = anchor_frame.shape[1]
    if explicit_indices is not None:
        occ = tuple(int(i) for i in explicit_indices)
        if len(occ) != nocc or len(set(occ)) != nocc:
            raise ValueError(f"anchor_occ_indices must contain exactly {nocc} distinct entries")
        if any(i < 0 or i >= nao for i in occ):
            raise IndexError("anchor_occ_indices contains an out-of-range column")
        vir = tuple(i for i in range(nao) if i not in set(occ))
        return occ, vir, {"mode": "explicit", "cutoff_gap": None, "populations": None}

    mode = selection.lower().replace("-", "_")
    populations: np.ndarray | None = None
    cutoff_gap: float | None = None
    if mode == "first":
        occ_set = set(range(nocc))
    elif mode in ("sad", "hf", "hf_population"):
        if mode == "sad":
            density = np.asarray(mf.get_init_guess(mol, key="atom"))
            mode = "sad"
        else:
            density = np.asarray(mf.make_rdm1())
            mode = "hf_population"
        if density.ndim == 3 and density.shape[0] == 2:
            density = density.sum(axis=0)
        if density.shape != (nao, nao):
            raise ValueError(f"Anchor-selection density has shape {density.shape}, expected {(nao, nao)}")
        density = _hermitize(density)
        populations = np.real(np.diag(anchor_frame.conj().T @ s @ density @ s @ anchor_frame)) / 2.0
        ranking = np.argsort(-populations, kind="stable")
        occ_set = set(int(i) for i in ranking[:nocc])
        if 0 < nocc < nao:
            cutoff_gap = float(populations[ranking[nocc - 1]] - populations[ranking[nocc]])
    else:
        raise ValueError("anchor_selection must be 'sad', 'hf', or 'first'")

    # Preserve atom/AO label order within each selected sector.  Population
    # ranking chooses membership only and does not become a second orbital gauge.
    occ = tuple(i for i in range(nao) if i in occ_set)
    vir = tuple(i for i in range(nao) if i not in occ_set)
    info = {
        "mode": mode,
        "cutoff_gap": cutoff_gap,
        "populations": None if populations is None else tuple(float(x) for x in populations),
    }
    return occ, vir, info


def _cross_diagnostics(
    cross: np.ndarray, singular_values: np.ndarray | None = None
) -> dict[str, Any]:
    dimension = cross.shape[1]
    if dimension == 0:
        return {
            "dimension": 0,
            "singular_values": (),
            "minimum_singular_value": 1.0,
            "maximum_principal_angle_degrees": 0.0,
            "condition_number": 1.0,
            "chart_distance": 0.0,
            "chordal_distance": 0.0,
            "procrustes_residual_squared": 0.0,
            "log_absolute_determinant": 0.0,
        }
    singular = (
        np.real(np.linalg.svd(cross, compute_uv=False))
        if singular_values is None
        else np.real(np.asarray(singular_values))
    )
    bounded = np.clip(singular, 0.0, 1.0)
    minimum = float(singular[-1])
    maximum = float(singular[0])
    angle = float(np.degrees(np.arccos(bounded[-1])))
    condition = float(np.inf if minimum == 0.0 else maximum / minimum)
    return {
        "dimension": dimension,
        "singular_values": tuple(float(x) for x in singular),
        "minimum_singular_value": minimum,
        "maximum_principal_angle_degrees": angle,
        "condition_number": condition,
        "chart_distance": float(np.sqrt(max(0.0, 1.0 - bounded[-1] * bounded[-1]))),
        "chordal_distance": float(np.sqrt(np.sum(np.maximum(0.0, 1.0 - bounded**2)))),
        "procrustes_residual_squared": float(max(0.0, 2.0 * dimension - 2.0 * np.sum(singular))),
        "log_absolute_determinant": float(np.sum(np.log(np.maximum(singular, np.finfo(float).tiny)))),
    }


def _block_cross_diagnostics(
    target: np.ndarray,
    anchor: np.ndarray,
    s: np.ndarray | None = None,
) -> dict[str, Any]:
    cross = target.conj().T @ anchor if s is None else target.conj().T @ s @ anchor
    return _cross_diagnostics(cross)


def _partition_cross_diagnostics(
    target_occ: np.ndarray,
    target_vir: np.ndarray,
    anchor_occ: np.ndarray,
    anchor_vir: np.ndarray,
    s: np.ndarray | None = None,
) -> dict[str, Any]:
    occupied = _block_cross_diagnostics(target_occ, anchor_occ, s)
    virtual = _block_cross_diagnostics(target_vir, anchor_vir, s)
    return {
        "occupied": occupied,
        "virtual": virtual,
        "minimum_singular_value": min(
            occupied["minimum_singular_value"], virtual["minimum_singular_value"]
        ),
        "maximum_principal_angle_degrees": max(
            occupied["maximum_principal_angle_degrees"],
            virtual["maximum_principal_angle_degrees"],
        ),
    }


def _least_change_block(
    target: np.ndarray,
    anchor: np.ndarray,
    s: np.ndarray,
    *,
    stage: str,
    sector: str,
    rank_tolerance: float,
    angle_warning_degrees: float,
    record_warning: Callable[[str], None],
    collect_diagnostics: bool,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    dimension = target.shape[1]
    if dimension == 0:
        diagnostic = (
            _cross_diagnostics(np.empty((0, 0))) if collect_diagnostics else None
        )
        return target.copy(), diagnostic
    cross = target.conj().T @ s @ anchor
    left, singular, right_h = np.linalg.svd(cross, full_matrices=False)
    minimum = float(np.real(singular[-1]))
    maximum_angle = float(
        np.degrees(np.arccos(np.clip(minimum, 0.0, 1.0)))
    )
    if minimum <= rank_tolerance:
        raise np.linalg.LinAlgError(
            f"No unique {stage} least-change map in the {sector} sector: "
            f"minimum principal cosine {minimum:.3e} <= {rank_tolerance:.3e}."
        )
    if maximum_angle > angle_warning_degrees:
        record_warning(
            f"{stage} {sector} maximum principal angle is "
            f"{maximum_angle:.2f} degrees "
            f"(minimum cosine {minimum:.3e}); the gauge is sensitive."
        )
    # The product left @ right_h is unique for nonsingular cross even when
    # individual singular vectors within degenerate clusters are not.
    polar = left @ right_h
    aligned = target @ polar
    diagnostic = None
    if collect_diagnostics:
        diagnostic = _cross_diagnostics(cross, singular)
        diagnostic["polar_unitarity_error"] = float(
            np.linalg.norm(polar.conj().T @ polar - np.eye(dimension))
        )
    return aligned, diagnostic


def _align_partition(
    target_occ: np.ndarray,
    target_vir: np.ndarray,
    anchor_occ: np.ndarray,
    anchor_vir: np.ndarray,
    s: np.ndarray,
    *,
    stage: str,
    rank_tolerance: float,
    angle_warning_degrees: float,
    record_warning: Callable[[str], None],
    collect_diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    aligned_occ, occupied = _least_change_block(
        target_occ,
        anchor_occ,
        s,
        stage=stage,
        sector="occupied",
        rank_tolerance=rank_tolerance,
        angle_warning_degrees=angle_warning_degrees,
        record_warning=record_warning,
        collect_diagnostics=collect_diagnostics,
    )
    aligned_vir, virtual = _least_change_block(
        target_vir,
        anchor_vir,
        s,
        stage=stage,
        sector="virtual",
        rank_tolerance=rank_tolerance,
        angle_warning_degrees=angle_warning_degrees,
        record_warning=record_warning,
        collect_diagnostics=collect_diagnostics,
    )
    diagnostic = None
    if collect_diagnostics:
        assert occupied is not None and virtual is not None
        diagnostic = {
            "occupied": occupied,
            "virtual": virtual,
            "minimum_singular_value": min(
                occupied["minimum_singular_value"], virtual["minimum_singular_value"]
            ),
            "maximum_principal_angle_degrees": max(
                occupied["maximum_principal_angle_degrees"],
                virtual["maximum_principal_angle_degrees"],
            ),
        }
    return aligned_occ, aligned_vir, diagnostic


def _obtain_sap_hamiltonian(
    mol: Any,
    mf: Any,
    *,
    sap_basis: str | dict[str, Any] | None,
    sap_hamiltonian: np.ndarray | Callable[[Any, Any], np.ndarray] | None,
) -> tuple[np.ndarray, str]:
    if sap_hamiltonian is not None:
        value = sap_hamiltonian(mol, mf) if callable(sap_hamiltonian) else sap_hamiltonian
        return np.asarray(value), "custom_hamiltonian"

    try:
        from pyscf import scf
        from pyscf.gto.basis import load as load_basis
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PySCF SAP support is unavailable") from exc
    if not hasattr(scf.hf, "make_sap"):
        raise RuntimeError(
            "This PySCF version has no scf.hf.make_sap; provide sap_hamiltonian explicitly"
        )
    source = getattr(mf, "sap_basis", "sapgrasplarge") if sap_basis is None else sap_basis
    internal_basis = _load_sap_basis_for_molecule(mol, source, load_basis)
    potential = scf.hf.make_sap(mol, sap_basis=internal_basis)
    # make_sap returns only the electronic screening potential.  The spectral
    # SAP Hamiltonian used by PySCF is h_core + V_SAP.
    source_label = source if isinstance(source, str) else "user_sap_basis_dict"
    # Reproduce PySCF's documented nonrelativistic SAP initial-guess operator.
    # In particular, do not inherit an overridden/decorated mf.get_hcore (X2C,
    # user embedding, and so on); callers wanting that operator can pass the
    # explicit sap_hamiltonian escape hatch.
    return np.asarray(scf.hf.get_hcore(mol)) + np.asarray(potential), source_label


def _load_sap_basis_for_molecule(mol: Any, source: Any, loader: Callable[..., Any]) -> dict[str, np.ndarray]:
    atom_labels = [entry[0] for entry in mol._atom]
    result: dict[str, np.ndarray] = {}
    for atom_index, atom_label in enumerate(atom_labels):
        if atom_label in result:
            continue
        pure_symbol = (
            mol.atom_pure_symbol(atom_index)
            if hasattr(mol, "atom_pure_symbol")
            else atom_label
        )
        if isinstance(source, str):
            loaded = loader(source, atom_label)
            if isinstance(loaded, dict):
                entry = loaded.get(atom_label, loaded.get(pure_symbol))
                if entry is None and len(loaded) == 1:
                    entry = next(iter(loaded.values()))
            else:
                entry = loaded
        elif isinstance(source, dict):
            entry = source.get(atom_label, source.get(pure_symbol))
            if entry is None:
                raise KeyError(f"No SAP basis entry for atom label {atom_label!r}")
        else:
            raise TypeError("sap_basis must be a PySCF basis name or dictionary")
        result[atom_label] = _coerce_sap_entry(entry, atom_label)
    return result


def _coerce_sap_entry(entry: Any, atom_label: str) -> np.ndarray:
    try:
        direct = np.asarray(entry, dtype=float)
    except (TypeError, ValueError):
        direct = np.empty((0, 0))
    if direct.ndim == 2 and direct.shape[1] == 2:
        return direct
    try:
        converted = np.asarray(entry[0][1:], dtype=float)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Could not convert SAP basis data for {atom_label!r}") from exc
    if converted.ndim != 2 or converted.shape[1] != 2:
        raise ValueError(f"SAP basis data for {atom_label!r} does not contain exponent/coefficient pairs")
    return converted


def _generalized_eigen_residual(
    h: np.ndarray,
    s: np.ndarray,
    coeff: np.ndarray,
    energy: np.ndarray,
) -> float:
    residual = h @ coeff - (s @ coeff) * energy[np.newaxis, :]
    scale = max(float(np.linalg.norm(h @ coeff)), float(np.linalg.norm(s @ coeff)), 1.0)
    return float(np.linalg.norm(residual) / scale)


def _projector_frobenius_error(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> float:
    dimension = a.shape[1]
    if dimension == 0:
        return 0.0
    overlap = a.conj().T @ s @ b
    # ||P_a-P_b||_F = sqrt(2) ||(I-P_a)b|| for two equally sized
    # orthonormal frames.  Forming the residual directly avoids catastrophic
    # cancellation in 2*k - 2*||a^H S b||_F^2 when the projectors agree.
    residual = b - a @ overlap
    residual_squared = float(np.real(np.trace(residual.conj().T @ s @ residual)))
    return float(np.sqrt(2.0 * max(0.0, residual_squared)))


def _definition_fingerprint(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:20]


def _input_descriptor(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return f"str:{value}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"callable:{module}.{qualname}"
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()[:20]
        return f"ndarray:{contiguous.shape}:{contiguous.dtype}:{digest}"
    if isinstance(value, dict):
        items = tuple(
            (repr(key), _input_descriptor(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return f"dict:{_definition_fingerprint(items)}"
    if isinstance(value, (list, tuple)):
        items = tuple(_input_descriptor(item) for item in value)
        return f"{type(value).__name__}:{_definition_fingerprint(items)}"
    return f"{type(value).__module__}.{type(value).__name__}:{_definition_fingerprint(value)}"


def _basis_fingerprint(mol: Any) -> str:
    atoms = tuple(
        (
            int(index),
            str(mol.atom_symbol(index)),
            int(mol.atom_charge(index)),
        )
        for index in range(int(mol.natm))
    )
    shells: list[Any] = []
    for shell in range(int(mol.nbas)):
        shells.append(
            (
                int(mol.bas_atom(shell)),
                int(mol.bas_angular(shell)),
                int(mol.bas_nprim(shell)),
                int(mol.bas_nctr(shell)),
                int(mol.bas_kappa(shell)),
                np.asarray(mol.bas_exp(shell), dtype=float).round(14).tolist(),
                np.asarray(mol.bas_ctr_coeff(shell), dtype=float).round(14).tolist(),
            )
        )
    identity = (
        atoms,
        tuple(shells),
        bool(mol.cart),
        tuple(str(label) for label in mol.ao_labels()),
        repr(getattr(mol, "_ecp", {})),
        repr(getattr(mol, "_pseudo", {})),
    )
    return _definition_fingerprint(identity)
