#!/usr/bin/env python3
"""
Compare many-body EVCont state gradients against finite differences.

The finite-difference reference differentiates the EVCont total energies with
respect to all nuclear Cartesian coordinates.  The analytic gradient is the
many-body state gradient returned by evcont.ab_initio_gradients_loewdin, using
the requested abstract-basis derivative.

split_procrustes is included in the setup notes but skipped for analytic
gradient comparison because its basis derivative is intentionally a placeholder.
"""

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
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad


def build_h_chain_from_coords(coords, basis="sto-3g"):
    """Create an H chain molecule from Cartesian coordinates in Bohr."""
    atom = [("H", tuple(coord)) for coord in coords]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def h_chain_coords(natom, spacing_bohr):
    """Linear H-chain coordinates in Bohr."""
    coords = np.zeros((natom, 3))
    coords[:, 0] = np.arange(natom) * spacing_bohr
    return coords


def finite_difference_state_gradients(cont, coords, basis, step, nroots):
    """Finite-difference EVCont state energies wrt all nuclear coordinates."""
    fd_grad = np.zeros((nroots, coords.shape[0], coords.shape[1]))

    for atom_id in range(coords.shape[0]):
        for xyz in range(coords.shape[1]):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[atom_id, xyz] += step
            coords_minus[atom_id, xyz] -= step

            e_plus, _ = cont.approximate_multistate(
                build_h_chain_from_coords(coords_plus, basis=basis),
                nroots=nroots,
            )
            e_minus, _ = cont.approximate_multistate(
                build_h_chain_from_coords(coords_minus, basis=basis),
                nroots=nroots,
            )
            fd_grad[:, atom_id, xyz] = (e_plus - e_minus) / (2.0 * step)

    return fd_grad


def build_continuation(abstract_basis, natom, basis, train_spacings, nroots):
    """Build one FCI continuation object in a chosen abstract basis."""
    cont = FCI_EVCont_obj(nroots=nroots, abstract_basis=abstract_basis)
    for spacing in train_spacings:
        mol = build_h_chain_from_coords(
            h_chain_coords(natom, spacing),
            basis=basis,
        )
        cont.append_to_rdms(mol)
    return cont


natom = 4
nroots = 2
basis = "sto-3g"
train_spacings = [1.2, 1.8]
test_spacings = np.linspace(1.25, 1.75, 7)
fd_step = 1.0e-4

analytic_bases = ["SAO", "meta_lowdin"]
skipped_bases = ["split_procrustes"]
state_labels = ["ground", "excited"]

errors = {
    abstract_basis: np.zeros((len(test_spacings), nroots))
    for abstract_basis in analytic_bases
}

for abstract_basis in analytic_bases:
    cont = build_continuation(
        abstract_basis,
        natom=natom,
        basis=basis,
        train_spacings=train_spacings,
        nroots=nroots,
    )

    for i_spacing, spacing in enumerate(test_spacings):
        coords = h_chain_coords(natom, spacing)
        mol = build_h_chain_from_coords(coords, basis=basis)

        _, analytic_grad = get_multistate_energy_with_grad(
            mol,
            cont.one_rdm,
            cont.two_rdm,
            cont.overlap,
            nroots=nroots,
            abstract_basis=abstract_basis,
        )
        fd_grad = finite_difference_state_gradients(
            cont,
            coords,
            basis=basis,
            step=fd_step,
            nroots=nroots,
        )

        diff = analytic_grad - fd_grad
        errors[abstract_basis][i_spacing] = np.linalg.norm(
            diff.reshape(nroots, -1),
            axis=1,
        )

        print(
            f"{abstract_basis:12s} spacing={spacing:.3f} "
            f"ground={errors[abstract_basis][i_spacing, 0]:.3e} "
            f"excited={errors[abstract_basis][i_spacing, 1]:.3e}"
        )

for abstract_basis in skipped_bases:
    print(
        f"{abstract_basis:12s} analytic gradients skipped: "
        "basis derivatives are not implemented yet."
    )

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(6.5, 6.0),
    constrained_layout=True,
)

for state_id, ax in enumerate(axes):
    for abstract_basis in analytic_bases:
        ax.semilogy(
            test_spacings,
            errors[abstract_basis][:, state_id],
            marker="o",
            label=abstract_basis,
        )
    ax.set_ylabel(f"{state_labels[state_id]} ||grad - FD||")
    ax.grid(ls=":", alpha=0.6)
    ax.legend()

axes[-1].set_xlabel("H-H spacing (Bohr)")
fig.suptitle("H4 FCI EVCont gradient finite-difference check")

out_path = Path(__file__).with_name("h4_fci_gradient_basis_fd.png")
fig.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
