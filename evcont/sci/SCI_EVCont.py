import glob
import os
import shutil
from collections import Counter

import numpy as np

from pyscf import mcscf

# DICE specific imports
import PostProcessStates
import shci

from evcont.electron_integral_utils import get_basis
from evcont.basis.basis_utils import basis_requires_reference, get_basis_reference
from evcont.low_rank_utils import reduce_2rdm, vectorize_lowrank, unpack_vectorized_lowrank

## Helper functions for reading SHCI transition RDMs from postprocessing outputs.
def read2RDM(fname):
    with open(fname) as f:
        lines = f.readlines()

    norbs = int(lines[0])
    rdm2 = np.zeros((norbs, norbs, norbs, norbs))
    for line in lines[1:]:
        tokens = line.split()
        val, i, j, k, l = float(tokens[4]), int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])
        rdm2[i, j, k, l] = val

    return rdm2.transpose([0, 2, 1, 3])

def read1RDM(fname):
    with open(fname) as f:
        lines = f.readlines()

    norbs = int(lines[0])
    rdm1 = np.zeros((norbs, norbs))
    for line in lines[1:]:
        tokens = line.split()
        val, i, j = float(tokens[2]), int(tokens[0]), int(tokens[1])
        rdm1[i, j] = val

    return rdm1.T

# Helper functions for computing and processing tRDMs from SHCI outputs, 
# including caching with geometry/state labels for later retrieval during continuation model construction.
def _cleanup_transition_outputs(parent_dir):
    for pattern in ("spatialRDM.*.*.txt", "spatial1RDM.*.*.txt"):
        for path in glob.glob(os.path.join(parent_dir, pattern)):
            os.remove(path)

def _state_label(run_i, local_i, run_j, nroots):
    if run_i == run_j:
        return run_i, (local_i % nroots)
    if local_i < nroots:
        return run_i, local_i
    return run_j, local_i - nroots

def _cache_name(parent_dir, kind, bra_run, bra_state, ket_run, ket_state):
    fname = f"{kind}.g{bra_run}.s{bra_state}.g{ket_run}.s{ket_state}.txt"
    return os.path.join(parent_dir, fname)


def cache_transition_rdms(run_prefix_i, run_prefix_j, mol, nroots, run_i, run_j, parent_dir):
    """Run SHCI postprocessing for one run-pair and cache tRDM files with geometry/state labels."""

    #os.makedirs(parent_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(parent_dir)
        _cleanup_transition_outputs(".")
        PostProcessStates.makeTransitionRDM(run_prefix_i, run_prefix_j, mol.nao, mol.nelectron)

        if run_i == run_j:
            local_range = range(nroots)
        else:
            local_range = range(2 * nroots)

        for local_j in local_range:
            for local_i in range(local_j, local_range.stop):
                src_rdm1 = f"spatial1RDM.{local_i}.{local_j}.txt"
                src_rdm2 = f"spatialRDM.{local_i}.{local_j}.txt"
                if not (os.path.exists(src_rdm1) and os.path.exists(src_rdm2)):
                    continue

                bra_run, bra_state = _state_label(run_i, local_i, run_j, nroots)
                ket_run, ket_state = _state_label(run_i, local_j, run_j, nroots)
                dst_rdm1 = _cache_name(parent_dir, "rdm1", bra_run, bra_state, ket_run, ket_state)
                dst_rdm2 = _cache_name(parent_dir, "rdm2", bra_run, bra_state, ket_run, ket_state)
                shutil.copyfile(src_rdm1, dst_rdm1)
                shutil.copyfile(src_rdm2, dst_rdm2)
    finally:
        os.chdir(prev_cwd)

# Main selected CI continuation class.
class SCI_EVCont_obj:
    """SCI continuation container that mirrors the FCI EVCont layout."""

    def __init__(
        self,
        comp_mol,
        cisolver=None,
        nroots=1,
        roots_train=None,
        sweep_epsilon=1.0e-2,
        initial_states=None,
        parent_dir=None,
        lowrank=False,
        abstract_basis="SAO",
        abstract_basis_ref=None,
        abstract_basis_ref_mol=None,
        abstract_basis_kwargs=None,
        **kwargs,
    ):
        if comp_mol is None:
            raise ValueError("comp_mol must be provided for SCI continuation")

        self.comp_mol = comp_mol
        self.cisolver = cisolver
        self.sweep_epsilon = sweep_epsilon
        if parent_dir is None:
            counts = Counter(comp_mol.atom_symbol(i) for i in range(comp_mol.natm))
            parts = []

            if "C" in counts:
                c_count = counts.pop("C")
                parts.append("C" if c_count == 1 else f"C{c_count}")
            if "H" in counts:
                h_count = counts.pop("H")
                parts.append("H" if h_count == 1 else f"H{h_count}")

            for symbol in sorted(counts):
                count = counts[symbol]
                parts.append(symbol if count == 1 else f"{symbol}{count}")

            parent_dir = "".join(parts) if parts else "molecule"

        self.parent_dir = os.path.abspath(parent_dir)
        os.makedirs(self.parent_dir, exist_ok=True)
        self.abstract_basis = abstract_basis
        self.abstract_basis_ref = abstract_basis_ref
        self.abstract_basis_ref_mol = abstract_basis_ref_mol
        self.abstract_basis_kwargs = dict(abstract_basis_kwargs or {})
        self.initial_states = initial_states
        if initial_states is None:
            self.initial_states = self._get_initial_states(comp_mol)

        self.nroots = nroots
        if roots_train is None:
            self.roots_train = list(range(nroots))
        else:
            self.roots_train = roots_train
            assert isinstance(roots_train, list)

        # Computational-basis geometry and global transformation.
        self._ensure_abstract_basis_reference(comp_mol)
        self.comp_basis_oao = self.get_abstract_basis(comp_mol)
        self.comp_basis = get_basis(comp_mol, basis_type="canonical")

        s_comp = comp_mol.intor("int1e_ovlp")
        self.global_trafo = np.einsum("ji,jk,kl->il", self.comp_basis, s_comp, self.comp_basis_oao)

        self.ens = []
        self.ens_nuc = []
        self.mol_index = []
        self.state_labels = []
        self.state_to_index = {}
        self.run_prefixes = []
        self.mols = []

        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

        self.lowrank = lowrank
        if lowrank:
            self.kwargs = kwargs

        self.diagonal_lr = None
        self.vecs_lowrank = {}

    def _ensure_abstract_basis_reference(self, mol):
        if not basis_requires_reference(self.abstract_basis):
            return
        if self.abstract_basis_ref is None:
            self.abstract_basis_ref = get_basis_reference(
                mol,
                basis_type=self.abstract_basis,
                **self.abstract_basis_kwargs,
            )
            self.abstract_basis_ref_mol = mol.copy()

    def get_abstract_basis(self, mol):
        self._ensure_abstract_basis_reference(mol)
        return get_basis(
            mol,
            basis_type=self.abstract_basis,
            basis_ref=self.abstract_basis_ref,
            basis_ref_mol=self.abstract_basis_ref_mol,
            **self.abstract_basis_kwargs,
        )

    def vectorize_lowrank(self, hermitian=True):
        vectorize_lowrank(self, hermitian=hermitian)

    def unpack_vectorized_lowrank(self):
        unpack_vectorized_lowrank(self)

    def transformed_basis(self, mol):
        """Return the local basis transformed into the stored computational basis."""
        basis_oao = self.get_abstract_basis(mol)
        # Match the reference workflow:
        # basis_i = einsum('ij,kj->ik', basis_OAO, trafo)
        return np.einsum("ij,kj->ik", basis_oao, self.global_trafo)

    def register_run(self, mol, prefix, energy=None):
        """Record one SHCI run for later pairwise postprocessing."""
        self.mols.append(mol)
        self.run_prefixes.append(prefix)

    def _get_initial_states(self, mol):
        """ Generate a list of initial states for SHCI from the Hartree-Fock configuration."""
        nmax = mol.nelectron
        hforder = [i for i in range(nmax)] # Occupied HF space
        # Excited determinants are generated by exciting electrons from the 
        # highest occupied orbital to the second/fourth lowest unoccupied orbital.
        hforder2 = [i for i in range(nmax)]
        hforder2[-1] = hforder[-1]+2
        hforder3 = [i for i in range(nmax)]
        hforder3[-1] = hforder[-1]+4
        initialStates=[hforder,hforder2,hforder3]
        return initialStates

    def _solve_shci_state(self, mol, prefix, initial_states=None, sweep_epsilon=None):
        # Run SHCI in the computational basis defined by comp_mol.
        basis = self.transformed_basis(mol)

        if sweep_epsilon is None:
            sweep_epsilon = self.sweep_epsilon
        if initial_states is None:
            initial_states = self.initial_states

        mc2 = mcscf.CASCI(mol, mol.nao, mol.nelec)
        mc2.fcisolver = shci.SHCI(mol)
        mc2.fcisolver.sweep_iter = [0]
        mc2.fcisolver.sweep_epsilon = [sweep_epsilon]
        #mc2.fcisolver.scratchDirectory = "."
        mc2.fcisolver.nroots = self.nroots
        mc2.fcisolver.davdisonTol = 1.0e-10
        os.makedirs(prefix, exist_ok=True)
        mc2.fcisolver.scratchDirectory = prefix
        if initial_states is not None:
            mc2.fcisolver.initialStates = initial_states

        # Following keyword currently gives the wrong tRDMs
        #mc2.fcisolver.extraline = ['Treversal 1'] # No triplet states

        # Enforce spin multiplicity (NOT WORKING WITH DICE)
        #mc2.fix_spin_(ss=0)

        e_all = mc2.kernel(basis)[0]
        if self.nroots == 1:
            e_all = [e_all]
        return e_all

    def _solve_shci_state(
        self,
        mol,
        prefix,
        initial_states=None,
        sweep_epsilon=None,
        sweep_iter=None,
        davidson_tol=1.0e-8,
        max_retries=2,
    ):
        basis = self.transformed_basis(mol)

        # Defaults
        if sweep_epsilon is None:
            sweep_epsilon = self.sweep_epsilon
        if initial_states is None:
            initial_states = self.initial_states

        # Normalize schedule inputs
        # Accept scalar or list for sweep_epsilon and sweep_iter.
        if np.isscalar(sweep_epsilon):
            # Black-box robust default annealing ladder ending at target epsilon.
            target = float(sweep_epsilon)
            eps_schedule = [max(1e-3, 20 * target), max(3e-4, 10 * target),
                            max(1e-4, 3 * target), target]
            # Make unique, monotone-decreasing
            eps_schedule = sorted(set(eps_schedule), reverse=True)
        else:
            eps_schedule = [float(x) for x in sweep_epsilon]

        if sweep_iter is None:
            # Cumulative iter schedule aligned with eps ladder
            iter_schedule = [4 * i for i in range(len(eps_schedule))]
        else:
            if np.isscalar(sweep_iter):
                iter_schedule = [int(sweep_iter) * i for i in range(len(eps_schedule))]
            else:
                iter_schedule = [int(x) for x in sweep_iter]

        if len(iter_schedule) != len(eps_schedule):
            raise ValueError("sweep_iter and sweep_epsilon must have the same length")

        # Ensure monotone nondecreasing iterations
        if any(j < i for i, j in zip(iter_schedule[:-1], iter_schedule[1:])):
            raise ValueError("sweep_iter must be nondecreasing")

        os.makedirs(prefix, exist_ok=True)

        # Retry strategy:
        # try with provided initialStates first; then cold-start if needed.
        trial_initial_states = [initial_states, None] if initial_states is not None else [None]

        last_err = None
        for attempt in range(min(max_retries, len(trial_initial_states))):
            try:
                mc2 = mcscf.CASCI(mol, mol.nao, mol.nelec)
                mc2.fcisolver = shci.SHCI(mol)

                mc2.fcisolver.nroots = self.nroots
                mc2.fcisolver.scratchDirectory = prefix

                mc2.fcisolver.sweep_iter = iter_schedule
                mc2.fcisolver.sweep_epsilon = eps_schedule

                # Important: correct spelling
                mc2.fcisolver.davidsonTol = davidson_tol

                # Deterministic/black-box friendly defaults
                mc2.fcisolver.nPTiter = 0  # variational only for stable continuation data

                init_try = trial_initial_states[attempt]
                if init_try is not None:
                    mc2.fcisolver.initialStates = init_try

                e_all = mc2.kernel(basis)[0]
                if self.nroots == 1:
                    e_all = [e_all]
                return e_all

            except Exception as err:
                last_err = err
                # Tighten schedule slightly on retry
                eps_schedule = [max(1e-6, 0.5 * e) for e in eps_schedule]
                davidson_tol = min(davidson_tol, 1.0e-9)
                continue

        raise RuntimeError(f"SHCI failed after retries at prefix={prefix}: {last_err}")

    def _resolve_prefix(self, prefix):
        if os.path.isabs(prefix):
            return prefix
        return os.path.join(self.parent_dir, prefix)

    def append_to_rdms(
        self,
        mol,
        shci_prefix=None,
        initial_states=None,
        sweep_epsilon=None,
    ):
        """Append a new training geometry and incrementally extend overlap/tRDM tensors."""

        run_new = len(self.run_prefixes)
        if shci_prefix is None:
            shci_prefix = f"run{run_new}"
        shci_prefix = self._resolve_prefix(shci_prefix)

        e_all = self._solve_shci_state(
            mol,
            shci_prefix,
            initial_states=initial_states,
            sweep_epsilon=sweep_epsilon,
        )

        self.register_run(mol, shci_prefix, energy=e_all)

        selected_states = [st for st in self.roots_train if st < len(e_all)]
        if len(selected_states) == 0:
            return

        old_nstate = 0 if self.overlap is None else self.overlap.shape[0]
        new_nstate = old_nstate + len(selected_states)

        overlap_new = np.ones((new_nstate, new_nstate))
        one_rdm_new = np.ones((new_nstate, new_nstate, mol.nao, mol.nao))
        if self.overlap is not None:
            overlap_new[:old_nstate, :old_nstate] = self.overlap
        if self.one_rdm is not None:
            one_rdm_new[:old_nstate, :old_nstate, :, :] = self.one_rdm

        if not self.lowrank:
            two_rdm_new = np.ones((new_nstate, new_nstate, mol.nao, mol.nao, mol.nao, mol.nao))
            if self.two_rdm is not None:
                two_rdm_new[:old_nstate, :old_nstate, :, :, :, :] = self.two_rdm
            diagonal_lr_new = None
        else:
            two_rdm_new = None
            diagonal_lr_new = np.ones((new_nstate, new_nstate, 3, mol.nao, mol.nao))
            if self.diagonal_lr is not None:
                diagonal_lr_new[:old_nstate, :old_nstate, :, :, :] = self.diagonal_lr

        run_new = len(self.run_prefixes) - 1
        for st in selected_states:
            self.state_labels.append((run_new, st))
            self.state_to_index[(run_new, st)] = old_nstate + selected_states.index(st)
            self.ens.append(e_all[st])
            self.ens_nuc.append(mol.energy_nuc())
            self.mol_index.append(run_new)

        def ingest_pair(run_i, run_j, same_run=False):
            cache_transition_rdms(
                self.run_prefixes[run_i],
                self.run_prefixes[run_j],
                self.mols[run_i],
                self.nroots,
                run_i,
                run_j,
                self.parent_dir,
            )
            local_limit = self.nroots if same_run else 2 * self.nroots
            for local_j in range(local_limit):
                for local_i in range(local_j, local_limit):
                    bra_run, bra_state = _state_label(run_i, local_i, run_j, self.nroots)
                    ket_run, ket_state = _state_label(run_i, local_j, run_j, self.nroots)
                    if bra_state not in self.roots_train or ket_state not in self.roots_train:
                        continue
                    if (bra_run, bra_state) not in self.state_to_index or (ket_run, ket_state) not in self.state_to_index:
                        continue

                    rdm1_path = _cache_name(self.parent_dir, "rdm1", bra_run, bra_state, ket_run, ket_state)
                    rdm2_path = _cache_name(self.parent_dir, "rdm2", bra_run, bra_state, ket_run, ket_state)
                    if not (os.path.exists(rdm1_path) and os.path.exists(rdm2_path)):
                        continue

                    rdm1 = read1RDM(rdm1_path)
                    rdm2 = read2RDM(rdm2_path)
                    rdm1 = np.einsum("...ij,ia,jb->...ab", rdm1, self.global_trafo, self.global_trafo, optimize="optimal")
                    rdm2 = np.einsum(
                        "...ijkl,ia,jb,kc,ld->...abcd",
                        rdm2,
                        self.global_trafo,
                        self.global_trafo,
                        self.global_trafo,
                        self.global_trafo,
                        optimize="optimal",
                    )

                    ib = self.state_to_index[(bra_run, bra_state)]
                    ik = self.state_to_index[(ket_run, ket_state)]
                    one_rdm_new[ib, ik, :, :] = rdm1
                    one_rdm_new[ik, ib, :, :] = rdm1.conj()
                    if not self.lowrank:
                        two_rdm_new[ib, ik, :, :, :, :] = rdm2
                        two_rdm_new[ik, ib, :, :, :, :] = rdm2.conj()
                    else:
                        ovlp = np.trace(rdm1) / self.comp_mol.nelectron
                        lowrank_vecs, diagonals, use_joint = reduce_2rdm(
                            rdm1,
                            rdm2,
                            ovlp,
                            mol=self.comp_mol,
                            train_en=e_all[0],
                            abstract_basis=self.abstract_basis,
                            basis_ref=self.abstract_basis_ref,
                            basis_kwargs=self.abstract_basis_kwargs,
                            **self.kwargs,
                        )
                        diagonal_lr_new[ib, ik, :, :, :] = diagonals
                        try:
                            diagonal_lr_new[ik, ib, :, :, :] = diagonals.conj()
                        except Exception:
                            diagonal_lr_new[ik, ib, :, :, :] = diagonals
                        self.vecs_lowrank[(ib, ik)] = (lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint)
                        self.vecs_lowrank[(ik, ib)] = (lowrank_vecs[0].conj(), lowrank_vecs[1].conj(), lowrank_vecs[2].conj(), use_joint)

            if same_run:
                # Same-run cache files are the standard self-RDM blocks.
                pass

        for old_run in range(run_new):
            ingest_pair(old_run, run_new, same_run=False)
        ingest_pair(run_new, run_new, same_run=True)

        self.overlap = np.einsum("abcc->ab", one_rdm_new) / self.comp_mol.nelectron
        self.one_rdm = one_rdm_new
        if not self.lowrank:
            self.two_rdm = two_rdm_new
        else:
            self.diagonal_lr = diagonal_lr_new
