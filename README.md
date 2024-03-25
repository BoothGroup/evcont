# EVCont
This package bundles different scripts and tools for the application of the eigenvector continuation from few variational *ab initio* states, as discussed in our upcoming manuscript: Y. Rath and G. H. Booth. Interpolating many-body wave functions for accelerated molecular dynamics on near-exact electronic surfaces [1]



## Installation
The project comes with a pyproject.toml.
The utility functions from the [evcont](./evcont) folder can be installed from the main folder with

```
pip install -e .
```

This also ensures that the core dependencies (numpy, scipy, pyscf) are installed.

Additional optional dependencies are not installed automatically and need to be installed manually if required.

Optional dependencies include:
- [block2](https://github.com/block-hczhai/block2-preview): required for the continuation from MPS
- [pygnme](https://github.com/BoothGroup/pygnme/blob/master/README.md?plain=1): required for the continuation from CAS states
- [dscribe](https://github.com/SINGROUP/dscribe): Required for GAP predictions

## Code

This repository is merely a collection of utility functions and scripts for the eigenvector continuation (in particular, interfacing different codes).
The [evcont](./evcont) folder contains common helper functions and bundles the main utilities (which can be installed).
The [scripts](./scripts) folder contains scripts generate the data from our manuscript [1].
These should also serve as a good first point of entry into the general functionality of the codebase.

The following scripts are included:
 - [scripts/PES_H_chain/H6_PES/H6_continuation.py](./scripts/PES_H_chain/H6_PES/H6_continuation.py): Prediction of the PES for a 6-atom H chain from different training points as depicted in Fig. (1) of [1]
 - [scripts/PES_H_chain/H10_PES/H10_continuation_3D_replacements.py](./scripts/PES_H_chain/H10_PES/H10_continuation_3D_replacements.py): Script to generate the data from Fig. (2) of [1], generating the EVCont, HF and GAP predictions of PES and force field for distorted 10-atom Hydrogen chains based on symmetrically stretched training configurations.
 - [scripts/MD/H2O](./scripts/MD/H2O): Scripts to perform the MD simulation of the water molecule (results shown in Figs. (3) and (6) of [1])
    - [scripts/MD/H2O/md_H2O_6_31G_DMRG_continuation.py](./scripts/MD/H2O/md_H2O_6_31G_DMRG_continuation.py): Runs the MD simulation for a water molecule in the 6-31G basis with a continuation from MPS.
    - [scripts/MD/H2O/md_H2O_vdz_CAS_continuation.py](./scripts/MD/H2O/md_H2O_vdz_CAS_continuation.py): Runs the MD simulation for a water molecule in the cc-pVDZ with a continuation from CAS states.
    - [scripts/MD/H2O/md_H2O_vtz_CAS_continuation.py](./scripts/MD/H2O/md_H2O_vtz_CAS_continuation.py): Runs the MD simulation for a water molecule in the cc-pVTZ with a continuation from CAS states.
    - [scripts/MD/H2O/md_H2O_6_31G_FCI.py](./scripts/MD/H2O/md_H2O_6_31G_FCI.py): Runs a reference MD simulation for a water molecule in the 6-31G basis with FCI.
    - [scripts/MD/H2O/md_H2O_vdz_CAS.py](./scripts/MD/H2O/md_H2O_vdz_CAS.py): Runs the MD simulation for a water molecule in the cc-pVDZ with CASCI.
    - [scripts/MD/H2O/md_H2O_vtz_CAS.py](./scripts/MD/H2O/md_H2O_vtz_CAS.py): Runs the MD simulation for a water molecule in the cc-pVTZ with CASCI.
    - [scripts/MD/H2O/evaluate_accuracy_6_31G.py](./scripts/MD/H2O/evaluate_accuracy_6_31G.py): Evaluates the energy and force along the converged trajectory in the 6-31G basis (for continuation and FCI), see Fig. (6) of [1].
    - [scripts/MD/H2O/evaluate_accuracy_vdz.py](./scripts/MD/H2O/evaluate_accuracy_vdz.py): Evaluates the energy and force along the converged trajectory in the cc-pVDZ basis (for continuation and CASCI), see Fig. (6) of [1].
    - [scripts/MD/H2O/evaluate_accuracy_vtz.py](./scripts/MD/H2O/evaluate_accuracy_vtz.py): Evaluates the energy and force along the converged trajectory in the cc-pVTZ basis (for continuation and CASCI), see Fig. (6) of [1].
 - [scripts/MD/H30](./scripts/MD/H30): Scripts to perform the MD simulation of the 1D hydrogen chain with 30 atoms in a minimal basis (results shown in Figs. (4) and (7) of [1])
    - [scripts/MD/H30/md_H30_evcont_from_DMRG.py](./scripts/MD/H30/md_H30_evcont_from_DMRG.py): MD simulation from an eigenvector continuation with MPS.
    - [scripts/MD/H30/md_H30_reference_DMRG_OAO.py](./scripts/MD/H30/md_H30_reference_DMRG_OAO.py): MD simulation with DMRG in the OAO basis.
    - [scripts/MD/H30/md_H30_evcont_from_DMRG_check_accuracy.py](./scripts/MD/H30/md_H30_evcont_from_DMRG_check_accuracy.py): Evaluate PES (DMRG reference + continuation) along the converged MD trajectory, see Fig. (7).
    - [scripts/MD/H30/md_H30_HF.py](./scripts/MD/H30/md_H30_HF.py): MD simulation with Hartree-Fock.
    - [scripts/MD/H30/md_H30_DFT.py](./scripts/MD/H30/md_H30_DFT.py): MD simulation with DFT (PBE exchange correlation function).
    - [scripts/MD/H30/md_H30_GAP.py](./scripts/MD/H30/md_H30_GAP.py): MD simulation with GAP predictions (using the same training data from a previous continuation run).
 - [scripts/MD/H2O-H3O+](./scripts/MD/H2O-H3O+): Scripts to perform the MD simulation of the Zundel cation complex in a 6-31G basis (results shown in Figs. (5) and (8) of [1])
    - [scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_continuation.py](./scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_continuation.py): Evaluate the PES, dipole moments, and atomic Mulliken charges predicted by the eigenvector continuation with a specified number of training points along the trajectories obtained a) with the restricted number of training points and b) the final continuation trajectory.
with all training points.
    - [scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_CCSD_final_continuation_trajectory.py](./scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_CCSD_final_continuation_trajectory.py): Evaluates the PES, dipole moments, and atomic Mulliken charges along the final continuation trajectory with CCSD. Requires the output of a preceding continuation run.
    - [scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_continuation.py](./scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_CCSD_final_continuation_trajectory.py): Evaluates the PES, dipole moments, and atomic Mulliken charges predicted by the eigenvector continuation with a specified number of training points along the trajectories obtained a) with the restricted number of training points and b) the final continuation trajectory with all training points. Requires the output of a preceding continuation run.
    - [scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_DFT_final_continuation_trajectory.py](./scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_DFT_final_continuation_trajectory.py): Evaluates the PES, dipole moments, and atomic Mulliken charges along the final continuation trajectory with DFT (B3LYP exchange correlation functional). Requires the output of a preceding continuation run.
    - [scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_HF_final_continuation_trajectory.py](./scripts/MD/H2O-H3O+/evaluate_dipole_moment_charges_HF_final_continuation_trajectory.py): Evaluates the PES, dipole moments, and atomic Mulliken charges along the final continuation trajectory with HF. Requires the output of a preceding continuation run.
    - [scripts/MD/H2O-H3O+/evaluate_energetics_training_points.py](./scripts/MD/H2O-H3O+/evaluate_energetics_training_points.py): Evaluates energetics for training points of Zundel cation MD simulation with DMRG, eigenvector continuation, HF, DFT, and CCSD. Requires the output of a preceding continuation run.
    - [scripts/MD/H2O-H3O+/md_H2O-H3O+_continuation_DMRG.py](./scripts/MD/H2O-H3O+/md_H2O-H3O+_continuation_DMRG.py): MD simulation from an eigenvector continuation with MPS.
    - [scripts/MD/H2O-H3O+/md_H2O-H3O+_HF.py](./scripts/MD/H2O-H3O+/md_H2O-H3O+_HF.py): MD simulation with Hartree-Fock, this also calculates the predicted dipole moment.
    - [scripts/MD/H2O-H3O+/md_H2O-H3O+_CCSD.py](./scripts/MD/H2O-H3O+/md_H2O-H3O+_CCSD.py): MD simulation with CCSD, this also calculates the predicted dipole moment.
    - [scripts/MD/H2O-H3O+/md_H2O-H3O+_DFT.py](./scripts/MD/H2O-H3O+/md_H2O-H3O+_DFT.py): MD simulation with DFT (B3LYP exchange correlation function), this also calculates the predicted dipole moment.


## Contact
Questions? Feel free to contact me via [yannic.rath@npl.co.uk](mailto:yannic.rath@npl.co.uk).

## Manuscript
[1] Y. Rath and G. H. Booth. Interpolating many-body wave functions for accelerated molecular dynamics on near-exact electronic surfaces
