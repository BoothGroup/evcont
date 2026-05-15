# EVCont
This package bundles different scripts and tools for the application of the eigenvector continuation from few variational *ab initio* states, as presented in: 
Rath, Y., & Booth, G. H. (2025) [1]
& Atalar, K., Rath, Y., Crespo-Otero, R. & Booth, G. H. (2024) [2]

The codebase also includes low-rank compression protocols for two-body reduced density matrices and functionality to infer from compressed states, as described in:
Atalar, K., Burton, H. G. A., Grüneis, A. & Booth, G. H. (2026) [3]

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
- [quantel](https://github.com/hgaburton/quantel): for alternative state-specific CASSCF solutions 
- [dscribe](https://github.com/SINGROUP/dscribe): Required for GAP predictions - as used in comparison scripts in [scripts](./scripts)

The codebase is interfaced to:
- [Newton-X](https://newtonx.org): for nonadiabatic molecular dynamics simulations 

which require separate installations. 

The [nx-interface](./nx-interface/) folder contains the driver script used to call this codebase within Newton-X workflows.


## Code

This repository is merely a collection of utility functions and scripts for the eigenvector continuation (in particular, interfacing different codes).
The [evcont](./evcont) folder contains common helper functions and bundles the main utilities (which can be installed).

The [examples](./examples) folder contains example scripts for demonstrating the general functionality of the codebase.

The [scripts](./scripts) folder contains scripts generate the data from our manuscript [1].
They also serve as example workflows for the codebase.

The following scripts are included:
 - [scripts/PES_H_chain/H6_PES/H6_continuation.py](./scripts/PES_H_chain/H6_PES/H6_continuation.py): Prediction of the PES for a 6-atom H chain from different training points as depicted in Fig. (1) of [1]
 - [scripts/PES_H_chain/H10_PES/H10_continuation_3D_replacements.py](./scripts/PES_H_chain/H10_PES/H10_continuation_3D_replacements.py): Script to generate the data from Fig. (2) of [1], generating the EVCont, HF and GAP predictions of PES and force field for distorted 10-atom Hydrogen chains based on symmetrically stretched training configurations.
 - [scripts/MD/H2O](./scripts/MD/H2O): Scripts to perform the MD simulation of the water molecule (results shown in Figs. (3) and (4) of [1])
 - [scripts/MD/H30](./scripts/MD/H30): Scripts to perform the MD simulation of the 1D hydrogen chain with 30 atoms in a minimal basis (results shown in Figs. (5) and (S1) of [1])
 - [scripts/MD/Zundel_thermodynamics](./scripts/MD/Zundel_thermodynamics): Scripts to perform multi-trajectory MD simulations of the Zundel cation complex in a 6-31G basis for the extraction of room temperature thermodynamics (results shown in Figs. (6), (S2), and (S3) of [1])
 - [scripts/MD/H2O-H3O+](./scripts/MD/H2O-H3O+): Scripts to perform the MD simulation of the Zundel cation complex in a 6-31G basis (results shown in Figs. (7) and (S4) of [1])


## Contact
Questions? Feel free to contact us via [kemal.atalar@kcl.ac.uk](mailto:kemal.atalar@kcl.ac.uk) or [yannic.rath@npl.co.uk](mailto:yannic.rath@npl.co.uk).

## Manuscripts
[1] Rath, Y., & Booth, G. H. (2025). Interpolating numerically exact many-body wave functions for accelerated molecular dynamics. *Nature Communications*, 16(1), 2005.

[2] Atalar, K., Rath, Y., Crespo-Otero, R. & Booth, G. H. (2024). Fast and accurate nonadiabatic molecular dynamics enabled through variational interpolation of correlated electron wavefunctions. *Faraday Discussions*, 254, 542-569.

[3] Atalar, K., Burton, H. G. A., Grüneis, A. & Booth, G. H. (2026). Low-rank compression of two-electron reduced density matrices. [arXiv:2605.11253](https://arxiv.org/abs/2605.11253)