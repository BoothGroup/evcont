"""Analytic derivatives of Lowdin and meta-Lowdin AO transforms."""

from __future__ import annotations

import numpy as np
from pyscf import lo
from pyscf.lo import nao


_DEFAULT_PRE_ORTH = object()


def orth_ao_derivative(
    mol,
    method: str = "meta_lowdin",
    pre_orth_ao=_DEFAULT_PRE_ORTH,
    s: np.ndarray | None = None,
    overlap_grad: np.ndarray | None = None,
    cutoff: float = 1.0e-15,
    degeneracy_tol: float = 1.0e-10,
    return_derivatives: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return a PySCF AO orthogonalization matrix and optionally its nuclear derivative.

    Parameters
    ----------
    mol
        PySCF molecule.
    method
        ``"meta_lowdin"``/``"meta-lowdin"`` for PySCF's meta-Lowdin AO
        localization or ``"lowdin"`` for the full Lowdin transform.
    pre_orth_ao
        For ``meta_lowdin``, the PySCF pre-orthogonalizer. The default is
        ``"ANO"``, matching ``lo.orth_ao(mol, "meta_lowdin")``. For
        ``lowdin``, the default is ``None``, giving the full AO-basis
        ``S^{-1/2}`` transform used by the reference code in this project.
    s
        AO overlap matrix. If omitted, it is computed from ``mol``.
    overlap_grad
        Optional overlap derivative with shape ``(nao, nao, natm, 3)``.
        If omitted, it is computed from ``int1e_ipovlp``.
    cutoff
        Minimum eigenvalue retained in each Lowdin inverse square root, matching
        the PySCF threshold. The derivative is only smooth when retained
        eigenvalues remain above this threshold.
    degeneracy_tol
        Relative eigenvalue-difference tolerance used in the divided
        differences of the Lowdin Frechet derivative.

    Returns
    -------
    C, dC
        ``C`` is the AO-to-orthogonal-AO coefficient matrix. ``dC`` has shape
        ``(nao, nao, natm, 3)``, where ``dC[:, :, ia, xyz]`` is the derivative
        with respect to displacement of atom ``ia`` along Cartesian component
        ``xyz``.

    Notes
    -----
    The Lowdin derivative is evaluated as the Frechet derivative of the matrix
    function ``S^{-1/2}`` using spectral divided differences. This is invariant
    to arbitrary rotations inside degenerate eigenspaces.
    """

    method_key = method.lower().replace("-", "_")
    if method_key not in {"lowdin", "meta_lowdin"}:
        raise ValueError("method must be 'lowdin' or 'meta_lowdin'")

    if pre_orth_ao is _DEFAULT_PRE_ORTH:
        pre_orth_ao = "ANO" if method_key == "meta_lowdin" else None

    if s is None:
        s = mol.intor_symmetric("int1e_ovlp")
    s = np.asarray(s)
    nao_nr = s.shape[0]

    if return_derivatives:
        if overlap_grad is None:
            inner_deriv = mol.intor("int1e_ipovlp", comp=3)
            d_s = np.zeros((mol.natm, 3, nao_nr, nao_nr), dtype=s.dtype)
            for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                d_s[ia, :, p0:p1, :] -= inner_deriv[:, p0:p1, :]
            d_s += d_s.swapaxes(-1, -2)
        else:
            d_s = np.asarray(overlap_grad).transpose(2, 3, 0, 1)

    def lowdin_factor(mat):
        mat = (np.asarray(mat) + np.asarray(mat).T.conj()) * 0.5
        eigvals, eigvecs = np.linalg.eigh(mat)
        active = eigvals > cutoff
        if not np.all(active):
            raise np.linalg.LinAlgError(
                "Lowdin derivative is not smooth for eigenvalues at or below "
                f"the PySCF cutoff {cutoff:g}; smallest eigenvalue is "
                f"{eigvals.min():.6e}."
            )

        f_vals = eigvals**-0.5
        lowdin = (eigvecs * f_vals) @ eigvecs.T.conj()

        diff = eigvals[:, None] - eigvals[None, :]
        scale = np.maximum(
            1.0, np.maximum(np.abs(eigvals[:, None]), np.abs(eigvals[None, :]))
        )
        separated = np.abs(diff) > degeneracy_tol * scale

        divided = np.empty_like(diff)
        divided[separated] = (
            f_vals[:, None] - f_vals[None, :]
        )[separated] / diff[separated]
        mean_eigvals = 0.5 * (eigvals[:, None] + eigvals[None, :])
        divided[~separated] = -0.5 * mean_eigvals[~separated] ** -1.5

        return {"matrix": lowdin, "eigvecs": eigvecs, "divided": divided}

    def lowdin_derivative(factor, d_mat):
        d_mat = (np.asarray(d_mat) + np.asarray(d_mat).T.conj()) * 0.5
        eigvecs = factor["eigvecs"]
        e_tilde = eigvecs.T.conj() @ d_mat @ eigvecs
        d_lowdin = eigvecs @ (factor["divided"] * e_tilde) @ eigvecs.T.conj()
        return (d_lowdin + d_lowdin.T.conj()) * 0.5

    if method_key == "lowdin":
        if pre_orth_ao is None:
            lowdin = lowdin_factor(s)
            c = lowdin["matrix"]
            if not return_derivatives:
                return c
            dc = np.empty((nao_nr, nao_nr, mol.natm, 3), dtype=s.dtype)
            for ia in range(mol.natm):
                for xyz in range(3):
                    dc[:, :, ia, xyz] = lowdin_derivative(lowdin, d_s[ia, xyz])
            return c, dc

        if not isinstance(pre_orth_ao, np.ndarray):
            pre_orth_ao = lo.orth.restore_ao_character(mol, pre_orth_ao)
        p = np.asarray(pre_orth_ao)
        s1 = p.T.conj() @ s @ p
        lowdin = lowdin_factor(s1)
        l1 = lowdin["matrix"]
        c = p @ l1
        if not return_derivatives:
            return c
        dc = np.empty((nao_nr, nao_nr, mol.natm, 3), dtype=s.dtype)
        for ia in range(mol.natm):
            for xyz in range(3):
                ds1 = p.T.conj() @ d_s[ia, xyz] @ p
                dl1 = lowdin_derivative(lowdin, ds1)
                dc[:, :, ia, xyz] = p @ dl1
        return c, dc

    if pre_orth_ao is None:
        p = np.eye(nao_nr, dtype=s.dtype)
    elif isinstance(pre_orth_ao, np.ndarray):
        p = np.asarray(pre_orth_ao)
    else:
        p = lo.orth.restore_ao_character(mol, pre_orth_ao)

    core_lst, val_lst, rydbg_lst = [
        np.asarray(x, dtype=int) for x in nao._core_val_ryd_list(mol)
    ]
    c = np.zeros((nao_nr, nao_nr), dtype=s.dtype)

    def lowdin_block(a):
        metric = a.T.conj() @ s @ a
        factor = lowdin_factor(metric)
        return factor, a @ factor["matrix"]

    def project_out(base, q):
        if q.shape[1] == 0:
            return base.copy()
        return base - q @ (q.T.conj() @ s @ base)

    def projected_base_derivative(base, q, dq, ds):
        if q.shape[1] == 0:
            return np.zeros_like(base)
        coeff = q.T.conj() @ s @ base
        d_coeff = dq.T.conj() @ s @ base + q.T.conj() @ ds @ base
        return -(dq @ coeff + q @ d_coeff)

    if core_lst.size:
        a_core = p[:, core_lst].copy()
        lowdin_core, c_core = lowdin_block(a_core)
        c[:, core_lst] = c_core
    else:
        a_core = p[:, :0]
        lowdin_core = None
        c_core = np.zeros((nao_nr, 0), dtype=s.dtype)

    val_base = p[:, val_lst].copy()
    if val_lst.size:
        a_val = project_out(val_base, c_core)
        lowdin_val, c_val = lowdin_block(a_val)
        c[:, val_lst] = c_val
    else:
        a_val = val_base
        lowdin_val = None
        c_val = np.zeros((nao_nr, 0), dtype=s.dtype)

    c_cv = np.hstack((c_core, c_val))
    rydbg_base = p[:, rydbg_lst].copy()
    if rydbg_lst.size:
        a_rydbg = project_out(rydbg_base, c_cv)
        lowdin_rydbg, c_rydbg = lowdin_block(a_rydbg)
        c[:, rydbg_lst] = c_rydbg
    else:
        a_rydbg = rydbg_base
        lowdin_rydbg = None
        c_rydbg = np.zeros((nao_nr, 0), dtype=s.dtype)

    if not return_derivatives:
        return c

    dc = np.zeros((nao_nr, nao_nr, mol.natm, 3), dtype=s.dtype)

    for ia in range(mol.natm):
        for xyz in range(3):
            ds = d_s[ia, xyz]

            if core_lst.size:
                ds_core = a_core.T.conj() @ ds @ a_core
                dl_core = lowdin_derivative(lowdin_core, ds_core)
                dc_core = a_core @ dl_core
                dc[:, core_lst, ia, xyz] = dc_core
            else:
                dc_core = c_core

            if val_lst.size:
                da_val = projected_base_derivative(val_base, c_core, dc_core, ds)
                ds_val = (
                    da_val.T.conj() @ s @ a_val
                    + a_val.T.conj() @ ds @ a_val
                    + a_val.T.conj() @ s @ da_val
                )
                dl_val = lowdin_derivative(lowdin_val, ds_val)
                dc_val = da_val @ lowdin_val["matrix"] + a_val @ dl_val
                dc[:, val_lst, ia, xyz] = dc_val
            else:
                dc_val = c_val

            if rydbg_lst.size:
                dc_cv = np.hstack((dc_core, dc_val))
                da_rydbg = projected_base_derivative(rydbg_base, c_cv, dc_cv, ds)
                ds_rydbg = (
                    da_rydbg.T.conj() @ s @ a_rydbg
                    + a_rydbg.T.conj() @ ds @ a_rydbg
                    + a_rydbg.T.conj() @ s @ da_rydbg
                )
                dl_rydbg = lowdin_derivative(lowdin_rydbg, ds_rydbg)
                dc_rydbg = da_rydbg @ lowdin_rydbg["matrix"] + a_rydbg @ dl_rydbg
                dc[:, rydbg_lst, ia, xyz] = dc_rydbg

    return c, dc
