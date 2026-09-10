"""Shared incremental orthonormalization of state transition RDMs.

If columns of ``X`` define old orthonormal states in a raw basis, then
``X.conj().T @ S @ X = I``.  For a newly appended raw state, this module forms
its projection ``p = X.conj().T @ S_old,new`` and residual norm
``r = S_new,new - p.conj() @ p``.  The new orthonormal state is the normalized
residual.  One- and two-body transition RDMs follow by linearity; solvers only
provide their raw RDM builder and low-rank storage policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from evcont.low_rank_utils import reconstruct_rdm2_joint, reduce_2rdm
from evcont.solver_evaluation import maintain_two_rdm_compression


def symmetrize_transition_rdm1(rdm1: np.ndarray) -> np.ndarray:
    """Hermitize a transition 1-RDM in its orbital indices."""

    return 0.5 * (rdm1 + np.asarray(rdm1).conj().T)


def adjoint_transition_rdm2(rdm2: np.ndarray) -> np.ndarray:
    """Return the bra/ket adjoint in the repository's 2-RDM convention."""

    return np.einsum("ijkl->jilk", np.asarray(rdm2).conj())


def symmetrize_transition_rdm2(rdm2: np.ndarray) -> np.ndarray:
    """Hermitize a transition 2-RDM in its paired orbital indices."""

    return 0.5 * (rdm2 + adjoint_transition_rdm2(rdm2))


def incremental_orthonormalizer(
    overlap: np.ndarray,
    old_transform: np.ndarray | None = None,
    lindep: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extend ``X`` by normalized projection of the final state in ``overlap``.

    Existing columns are preserved, unlike a new symmetric ``S**-1/2``.  This
    makes the transformation ordered, but avoids rotating all previous RDMs.
    """

    overlap = np.asarray(overlap)
    nstate = overlap.shape[0]
    if overlap.shape != (nstate, nstate):
        raise ValueError("overlap must be square")
    if nstate == 1:
        residual = float(np.real(overlap[0, 0]))
        projection = np.empty(0, dtype=overlap.dtype)
    else:
        old_transform = np.asarray(old_transform)
        if old_transform.shape != (nstate - 1, nstate - 1):
            raise ValueError("old_transform has the wrong shape")
        # p_a = <old orthonormal state a | new raw state>.
        projection = old_transform.T.conj() @ overlap[:-1, -1]
        # Squared norm after removing the projection: the Schur complement.
        residual = float(np.real(overlap[-1, -1] - np.vdot(projection, projection)))

    if not np.isfinite(residual) or residual <= lindep:
        raise np.linalg.LinAlgError(
            "new training state is linearly dependent after overlap projection: "
            f"Schur complement {residual:.3e} <= {lindep:.3e}"
        )
    if nstate == 1:
        transform = np.asarray([[1.0 / np.sqrt(residual)]], dtype=overlap.dtype)
    else:
        transform = np.zeros((nstate, nstate), dtype=overlap.dtype)
        transform[:-1, :-1] = old_transform
        # |new orthogonal> = (|new raw> - |old orthogonal> p) / sqrt(r).
        transform[:-1, -1] = -(old_transform @ projection) / np.sqrt(residual)
        transform[-1, -1] = 1.0 / np.sqrt(residual)
    return transform, projection, residual


@dataclass(frozen=True)
class OrthonormalRDMUpdate:
    """State-space arrays produced by one incremental update."""

    raw_one_rdm: np.ndarray
    raw_overlap: np.ndarray
    transform: np.ndarray
    one_rdm: np.ndarray
    overlap: np.ndarray


class IncrementalRDMOrthonormalizer:
    """Incrementally transform 1- and 2-RDMs into an orthonormal state basis.

    Solvers provide callbacks for constructing raw transition RDMs, restoring
    old 2-RDMs, and storing each new orthonormal 2-RDM. This keeps amplitude
    conventions and compression policies outside the shared transformation.
    """

    def __init__(self, electron_count: int, lindep: float = 1.0e-12):
        if electron_count <= 0:
            raise ValueError("electron_count must be positive")
        if lindep <= 0.0:
            raise ValueError("lindep must be positive")
        self.electron_count = electron_count
        self.lindep = lindep

    def append(
        self,
        *,
        nstate: int,
        norb: int,
        raw_one_rdm: np.ndarray | None,
        old_transform: np.ndarray | None,
        build_raw_pair: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
        reconstruct_old_rdm2: Callable[[int, int], np.ndarray],
        store_new_pair: Callable[[int, int, np.ndarray, np.ndarray, complex], None],
    ) -> OrthonormalRDMUpdate:
        """Append the final raw state and emit only its orthonormal 2-RDM pairs.

        ``build_raw_pair(i, n)`` supplies the solver-specific raw transition
        RDMs. ``reconstruct_old_rdm2(a, b)`` exposes previous orthonormal pairs,
        and ``store_new_pair`` consumes each newly transformed pair, typically
        by compressing it immediately.
        """

        if nstate <= 0:
            raise ValueError("nstate must be positive")
        nold = nstate - 1
        if nold:
            old_transform = np.asarray(old_transform)
            if old_transform.shape != (nold, nold):
                raise ValueError("old_transform has the wrong shape")
        if raw_one_rdm is None:
            if nold:
                raise RuntimeError("raw transition 1-RDMs are unavailable")
            expanded_one_rdm = np.zeros((1, 1, norb, norb))
        else:
            raw_one_rdm = np.asarray(raw_one_rdm)
            if raw_one_rdm.shape != (nold, nold, norb, norb):
                raise RuntimeError("raw transition 1-RDM state count is inconsistent")
            expanded_one_rdm = np.zeros(
                (nstate, nstate, norb, norb), dtype=raw_one_rdm.dtype
            )
            expanded_one_rdm[:nold, :nold] = raw_one_rdm

        # C_a = sum_i X^*_{ia} Gamma_{i,n}; only new-state raw 2-RDMs are built.
        transformed_cross = [
            np.zeros((norb, norb, norb, norb), dtype=expanded_one_rdm.dtype)
            for _ in range(nold)
        ]
        for raw_bra in range(nold):
            rdm1, rdm2 = build_raw_pair(raw_bra, nold)
            expanded_one_rdm[raw_bra, nold] = rdm1
            expanded_one_rdm[nold, raw_bra] = rdm1.conj().T
            for orthogonal_bra in range(nold):
                coefficient = old_transform[raw_bra, orthogonal_bra].conj()
                if coefficient != 0.0:
                    transformed_cross[orthogonal_bra] += coefficient * rdm2

        rdm1_nn, rdm2_nn = build_raw_pair(nold, nold)
        expanded_one_rdm[nold, nold] = rdm1_nn
        # Tr(gamma_ij) / N is the state overlap because N-hat is fixed.
        raw_overlap = (
            np.einsum("abcc->ab", expanded_one_rdm, optimize="optimal")
            / self.electron_count
        )
        raw_overlap = 0.5 * (raw_overlap + raw_overlap.T.conj())
        transform, projection, residual = incremental_orthonormalizer(
            raw_overlap,
            old_transform=old_transform,
            lindep=self.lindep,
        )
        # 1-RDMs are small enough to transform as the full X^dagger gamma X.
        one_rdm = np.einsum(
            "ia,ijpq,jb->abpq",
            transform.conj(),
            expanded_one_rdm,
            transform,
            optimize="optimal",
        )
        overlap = np.einsum("abcc->ab", one_rdm, optimize="optimal")
        overlap /= self.electron_count
        overlap = 0.5 * (overlap + overlap.T.conj())

        if nstate == 1:
            store_new_pair(0, 0, one_rdm[0, 0], rdm2_nn / residual, overlap[0, 0])
        else:
            sqrt_residual = np.sqrt(residual)
            projected_bra_new = np.zeros_like(rdm2_nn)
            projected_old_old = np.zeros_like(rdm2_nn)
            for bra in range(nold):
                projected_bra_new += projection[bra].conj() * transformed_cross[bra]
                projected_ket = np.zeros_like(rdm2_nn)
                # D_a = sum_b p_b Gamma_ab removes the old-space ket component.
                for ket in range(nold):
                    if projection[ket] != 0.0:
                        projected_ket += projection[ket] * reconstruct_old_rdm2(
                            bra, ket
                        )
                projected_old_old += projection[bra].conj() * projected_ket
                orthogonal_cross = symmetrize_transition_rdm2(
                    (transformed_cross[bra] - projected_ket) / sqrt_residual
                )
                store_new_pair(
                    bra,
                    nold,
                    one_rdm[bra, nold],
                    orthogonal_cross,
                    overlap[bra, nold],
                )

            # Expand <new raw - P|Gamma|new raw - P> / r.
            orthogonal_nn = (
                rdm2_nn
                - projected_bra_new
                - adjoint_transition_rdm2(projected_bra_new)
                + projected_old_old
            ) / residual
            store_new_pair(
                nold,
                nold,
                one_rdm[nold, nold],
                symmetrize_transition_rdm2(orthogonal_nn),
                overlap[nold, nold],
            )

        return OrthonormalRDMUpdate(
            raw_one_rdm=expanded_one_rdm,
            raw_overlap=raw_overlap,
            transform=transform,
            one_rdm=one_rdm,
            overlap=overlap,
        )


class RDMOrthonormalizationMixin:
    """Reusable low-rank transition-RDM orthonormalization behavior.

    Solvers share ``states`` and ``train_energies`` attributes and implement
    only their raw transition-RDM construction.
    """

    def _initialize_rdm_orthonormalization(self) -> None:
        self.raw_one_rdm = None
        self.raw_overlap = None
        self.state_transform = None
        self._lowrank_anchor_rdm2 = None
        if self.lowrank and self.lowrank_orthonormalize:
            if self.hermitise != "both":
                raise ValueError("lowrank_orthonormalize requires hermitise='both'")
            if self.lowrank_kwargs.get("truncation_style") == "ham_en":
                raise ValueError(
                    "ham_en truncation is undefined for orthonormalized "
                    "off-diagonal transition RDMs; use truncation_style='ham'"
                )

    def _raw_transition_rdms(
        self, bra: int, ket: int, mol
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _transition_rdms(
        self, bra: int, ket: int, mol
    ) -> tuple[np.ndarray, np.ndarray]:
        rdm1, rdm2 = self._raw_transition_rdms(bra, ket, mol)
        rdm1_reverse, rdm2_reverse = self._raw_transition_rdms(ket, bra, mol)
        if self.hermitise == "both":
            rdm1 = 0.5 * (rdm1 + rdm1_reverse.conj().T)
            rdm2 = 0.5 * (rdm2 + adjoint_transition_rdm2(rdm2_reverse))
        elif self.hermitise != "none":
            raise ValueError("hermitise must be 'both' or 'none'")

        rdm1 = symmetrize_transition_rdm1(rdm1)
        rdm2 = symmetrize_transition_rdm2(rdm2)
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
        return rdm1, rdm2

    def _lowrank_basis_kwargs(self) -> dict:
        """Provide the reference data needed by Hamiltonian-aware truncation."""

        basis_kwargs = dict(self.abstract_basis_kwargs)
        if self.abstract_basis_ref_mol is not None:
            basis_kwargs.setdefault("basis_ref_mol", self.abstract_basis_ref_mol)
        if self._basis_name == "split_procrustes":
            ref_mf = getattr(self, "abstract_basis_ref_mf", None)
            if ref_mf is not None:
                basis_kwargs.setdefault("ref_mf", ref_mf)
        return basis_kwargs

    def _compress_orthonormal_rdm2(
        self,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        overlap: complex,
        mol,
    ) -> tuple[tuple, np.ndarray | None]:
        """Compress one new pair using the common EVCont low-rank format."""

        reduction_options = dict(self.lowrank_kwargs)
        if reduction_options.get("truncation_style") in {"ham", "ham_en"}:
            safety = float(self.kwargs.get("orthonormal_ham_safety", 1.0))
            if not 0.0 < safety <= 1.0:
                raise ValueError("orthonormal_ham_safety must lie in (0, 1]")
            reduction_options["ham_thr"] = (
                reduction_options.get("ham_thr", 1.0e-3) * safety
            )
        vectors, diagonals, joint = reduce_2rdm(
            rdm1,
            rdm2,
            overlap,
            mol=mol,
            train_en=self.train_energies[0],
            abstract_basis=self.abstract_basis,
            basis_ref=self.abstract_basis_ref,
            basis_kwargs=self._lowrank_basis_kwargs(),
            **reduction_options,
        )
        values, left, right = vectors
        return (values, left, right, joint), diagonals

    def _reconstruct_orthonormal_rdm2(self, bra: int, ket: int) -> np.ndarray:
        """Restore one old pair for the incremental projection algebra."""

        # Keep the first exact pair to avoid seeding every later state with its
        # compression error; all other old pairs are reconstructed on demand.
        if bra == ket == 0 and self._lowrank_anchor_rdm2 is not None:
            return self._lowrank_anchor_rdm2.copy()
        values, left, right, joint = self.vecs_lowrank[bra, ket]
        diagonals = None if self.diagonal_lr is None else self.diagonal_lr[bra, ket]
        return reconstruct_rdm2_joint((values, left, right), diagonals, joint=joint)

    def _store_lowrank_pair(
        self, factors: tuple, diagonals: np.ndarray | None, bra: int, ket: int
    ) -> None:
        """Store a pair and derive its reverse by bra/ket Hermiticity."""

        values, left, right, joint = factors
        self.vecs_lowrank[bra, ket] = values, left, right, joint
        if bra != ket:
            self.vecs_lowrank[ket, bra] = (
                values.conj(),
                left.conj(),
                right.conj(),
                joint,
            )
        if diagonals is not None:
            self.diagonal_lr[bra, ket] = diagonals
            if bra != ket:
                self.diagonal_lr[ket, bra] = diagonals.conj()

    def _build_orthonormal_lowrank_transition_rdms(self, mol) -> None:
        """Orchestrate one incremental update after a solver appends a state."""

        nstate = len(self.states)
        if nstate == 0:
            raise ValueError("No states have been appended")
        nold = nstate - 1
        norb = mol.nao
        electron_count = mol.nelectron
        lindep = float(
            self.kwargs.get(
                "orthonormalization_lindep",
                self.kwargs.get("lindep", 1.0e-12),
            )
        )

        if self.diagonal_lr is not None or self.lowrank_kwargs.get("save_diag", False):
            dtype = float if self.raw_one_rdm is None else self.raw_one_rdm.dtype
            old_diagonal_lr = self.diagonal_lr
            self.diagonal_lr = np.zeros((nstate, nstate, 3, norb, norb), dtype=dtype)
            if old_diagonal_lr is not None:
                self.diagonal_lr[:nold, :nold] = old_diagonal_lr
        else:
            self.diagonal_lr = None

        def store_new_pair(bra, ket, rdm1, rdm2, overlap):
            if bra == ket == 0:
                self._lowrank_anchor_rdm2 = rdm2.copy()
            factors, diagonals = self._compress_orthonormal_rdm2(
                rdm1, rdm2, overlap, mol
            )
            self._store_lowrank_pair(factors, diagonals, bra, ket)

        update = IncrementalRDMOrthonormalizer(electron_count, lindep=lindep).append(
            nstate=nstate,
            norb=norb,
            raw_one_rdm=self.raw_one_rdm,
            old_transform=self.state_transform,
            build_raw_pair=lambda bra, ket: self._transition_rdms(bra, ket, mol),
            reconstruct_old_rdm2=self._reconstruct_orthonormal_rdm2,
            store_new_pair=store_new_pair,
        )
        self.raw_one_rdm = update.raw_one_rdm
        self.raw_overlap = update.raw_overlap
        self.state_transform = update.transform
        self.one_rdm = update.one_rdm
        self.two_rdm = None
        self.lowrank_vectorized = None
        self.diagonal_vectorized = None
        self.overlap = update.overlap

    @maintain_two_rdm_compression
    def build_transition_rdms(self, mol) -> None:
        """Build transition RDMs in the configured state representation."""

        if self.lowrank and self.lowrank_orthonormalize:
            self._build_orthonormal_lowrank_transition_rdms(mol)
        else:
            self._build_transition_rdms_pairwise(mol)
