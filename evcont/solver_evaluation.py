"""Common evaluation methods for eigenvector-continuation solver objects."""

from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_abstract_basis,
    approximate_multistate_lowrank_OAO,
)
from evcont.ab_initio_gradients_loewdin import (
    get_lowrank_en_with_grad_and_NAC,
    get_multistate_energy_with_grad,
    get_multistate_energy_with_grad_and_NAC,
)
from evcont.low_rank_utils import vectorize_lowrank


class EVContEvaluationMixin:
    """Evaluate a continuation object using its stored representation settings."""

    _lowrank_evaluation_keys = {
        "Jdiag_only",
        "sao_diag",
        "density_fit",
        "df_basis",
        "df_response",
        "hermitian",
        "lindep",
    }

    def _resolve_nroots(self, nroots):
        return self.nroots if nroots is None else nroots

    def _evaluation_basis_kwargs(self):
        basis_kwargs = dict(getattr(self, "abstract_basis_kwargs", {}) or {})
        basis_ref = getattr(self, "abstract_basis_ref", None)
        if basis_ref is not None:
            basis_kwargs.setdefault("basis_ref", basis_ref)
        basis_ref_mol = getattr(self, "abstract_basis_ref_mol", None)
        if basis_ref_mol is not None:
            basis_kwargs.setdefault("basis_ref_mol", basis_ref_mol)
        ref_mf = getattr(self, "abstract_basis_ref_mf", None)
        if ref_mf is not None and getattr(self, "_basis_name", None) == "split_procrustes":
            basis_kwargs.setdefault("ref_mf", ref_mf)
        return basis_kwargs

    def _lowrank_evaluation_kwargs(self, kwargs):
        options = {
            key: value
            for key, value in getattr(self, "kwargs", {}).items()
            if key in self._lowrank_evaluation_keys
        }
        options.update(kwargs)
        return options

    def _vectorized_lowrank(self, hermitian=True):
        vectorized = getattr(self, "lowrank_vectorized", None)
        if vectorized is None or vectorized.get("ntrain") != self.overlap.shape[0]:
            vectorize_lowrank(self, hermitian=hermitian)
        return self.lowrank_vectorized, getattr(self, "diagonal_vectorized", None)

    def get_en(self, mol, nroots=None, **kwargs):
        """Return continuation energies and subspace eigenvectors."""
        nroots = self._resolve_nroots(nroots)
        basis_kwargs = self._evaluation_basis_kwargs()

        if getattr(self, "lowrank", False):
            options = self._lowrank_evaluation_kwargs(kwargs)
            lowrank_vecs, diagonals = self._vectorized_lowrank(
                hermitian=options.get("hermitian", True)
            )
            basis_ref = basis_kwargs.pop("basis_ref", None)
            return approximate_multistate_lowrank_OAO(
                mol,
                self.one_rdm,
                lowrank_vecs,
                diagonals,
                self.overlap,
                nroots=nroots,
                abstract_basis=self.abstract_basis,
                basis_ref=basis_ref,
                basis_kwargs=basis_kwargs or None,
                **options,
            )

        basis_ref = basis_kwargs.pop("basis_ref", None)
        return approximate_multistate_abstract_basis(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=nroots,
            abstract_basis=self.abstract_basis,
            basis_ref=basis_ref,
            **basis_kwargs,
            **kwargs,
        )

    def get_en_with_grad(self, mol, nroots=None, **kwargs):
        """Return continuation energies and nuclear gradients."""
        if getattr(self, "lowrank", False):
            return_coefficients = kwargs.get("return_coefficients", False)
            kwargs["return_coefficients"] = return_coefficients
            return_rdms = kwargs.get("return_rdms", False)
            result = self.get_en_with_grad_and_NAC(mol, nroots=nroots, **kwargs)
            output = result[:3] if return_coefficients else result[:2]
            if return_rdms:
                output += (result[-1],)
            return output

        return get_multistate_energy_with_grad(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=self._resolve_nroots(nroots),
            abstract_basis=self.abstract_basis,
            basis_kwargs=self._evaluation_basis_kwargs() or None,
            **kwargs,
        )

    def get_en_with_grad_and_NAC(self, mol, nroots=None, **kwargs):
        """Return eigenvectors, energies, gradients, and nonadiabatic couplings."""
        nroots = self._resolve_nroots(nroots)
        basis_kwargs = self._evaluation_basis_kwargs()

        if getattr(self, "lowrank", False):
            options = self._lowrank_evaluation_kwargs(kwargs)
            lowrank_vecs, diagonals = self._vectorized_lowrank(
                hermitian=options.get("hermitian", True)
            )
            return get_lowrank_en_with_grad_and_NAC(
                mol,
                self.one_rdm,
                self.overlap,
                lowrank_vecs,
                diagonals=diagonals,
                nroots=nroots,
                abstract_basis=self.abstract_basis,
                basis_kwargs=basis_kwargs or None,
                **options,
            )

        return get_multistate_energy_with_grad_and_NAC(
            mol,
            self.one_rdm,
            self.two_rdm,
            self.overlap,
            nroots=nroots,
            abstract_basis=self.abstract_basis,
            basis_kwargs=basis_kwargs or None,
            **kwargs,
        )
