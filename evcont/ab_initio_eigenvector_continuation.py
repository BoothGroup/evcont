import numpy as np

import sys

from scipy.linalg import eigh, eig

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.low_rank_utils import lowrank_hamiltonian

from pyscf.lib import safe_eigh

def approximate_ground_state(h1, h2, one_RDM, two_RDM, S, hermitian=True):
    """
    Returns the electronic ground state approximation from solving the generalised
    eigenvalue problem defined via the one- and two-body transition RDMs.

    Args:
        h1 (np.ndarray): One-electron integrals.
        h2 (np.ndarray): Two-electron integrals.
        one_RDM (np.ndarray): One-body t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM.
        S (np.ndarray): Overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        Tuple[float, np.ndarray]: Energy approximation and ground state approximation.
    """
    # Calculate the Hamiltonian matrix
    H = np.einsum("ijkl,kl->ij", one_RDM, h1, optimize="optimal") + 0.5 * np.einsum(
        "ijklmn,klmn->ij", two_RDM, h2, optimize="optimal"
    )

    if hermitian is True:
        # Solve the generalized eigenvalue problem for Hermitian Hamiltonian
        #vals, vecs = eigh(H, S)
        vals, vecs, _ = safe_eigh(H, S)
    else:
        # Solve the generalized eigenvalue problem for non-Hermitian Hamiltonian
        vals, vecs = eig(H, S)

    # Filter out imaginary eigenvalues
    valid_vals = abs(vals.imag) < 1.0e-5

    # Find the index of the minimum GS eigenvalue
    argmin = np.argmin(vals[valid_vals].real)

    # Get the energy approximation and ground state approximation
    en_approx = vals[valid_vals][argmin].real
    gs_approx = vecs[:, valid_vals][:, argmin].real

    return en_approx, gs_approx


def approximate_multistate_lowrank(mol, one_RDM, cum_diagonal, lowrank_vecs, S, nroots=1, hermitian=True):
    """
    Returns multiple approximate electronic states from solving the generalised
    eigenvalue problem defined via the one- and two-body transition RDMs.

    Args:
        h1 (np.ndarray): One-electron integrals.
        h2 (np.ndarray): Two-electron integrals.
        one_RDM (np.ndarray): One-body t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        S (np.ndarray): Overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        Tuple[float, np.ndarray]: Energy approximation and ground state approximation.
    """
    # Calculate the Hamiltonian matrix
    H = lowrank_hamiltonian(mol, one_RDM, S, cum_diagonal, lowrank_vecs)

    if hermitian is True:
        # Solve the generalized eigenvalue problem for Hermitian Hamiltonian
        #vals, vecs = eigh(H, S)
        vals, vecs, _ = safe_eigh(H, S)
        
    else:
        # Solve the generalized eigenvalue problem for non-Hermitian Hamiltonian
        vals, vecs = eig(H, S)

    # Filter out imaginary eigenvalues
    valid_vals = abs(vals.imag) < 1.0e-5
    
    # Make sure nroots isn't higher than available eigenstates
    assert vals[valid_vals].shape[0] >= nroots

    # Find the index of the minimum GS eigenvalue
    argroots = np.argsort(vals[valid_vals].real)[:nroots]

    # Get the energy approximation and ground state approximation
    en_approx = vals[valid_vals][argroots].real
    evec_approx = vecs[:, valid_vals][:, argroots].real.T

    return en_approx, evec_approx

def approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=1, hermitian=True):
    """
    Returns multiple approximate electronic states from solving the generalised
    eigenvalue problem defined via the one- and two-body transition RDMs.

    Args:
        h1 (np.ndarray): One-electron integrals.
        h2 (np.ndarray): Two-electron integrals.
        one_RDM (np.ndarray): One-body t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        S (np.ndarray): Overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        Tuple[float, np.ndarray]: Energy approximation and ground state approximation.
    """
    # Calculate the Hamiltonian matrix
    H = np.einsum("ijkl,kl->ij", one_RDM, h1, optimize="optimal") + 0.5 * np.einsum(
        "ijklmn,klmn->ij", two_RDM, h2, optimize="optimal"
    )

    #print('  Hamiltonian')
    #print(H.tolist())
    #plt.figure()
    #sb.heatmap(H, annot=True)
    #plt.savefig('hamiltonian_%i.png'%(rdm_computation))
    #print('  Overlap')
    #print(S.tolist())
    
    if hermitian is True:
        # Solve the generalized eigenvalue problem for Hermitian Hamiltonian
        #vals, vecs = eigh(H, S)
        vals, vecs, _ = safe_eigh(H, S)
        
    else:
        # Solve the generalized eigenvalue problem for non-Hermitian Hamiltonian
        vals, vecs = eig(H, S)

    # Filter out imaginary eigenvalues
    valid_vals = abs(vals.imag) < 1.0e-5
    
    # Make sure nroots isn't higher than available eigenstates
    assert vals[valid_vals].shape[0] >= nroots

    # Find the index of the minimum GS eigenvalue
    argroots = np.argsort(vals[valid_vals].real)[:nroots]

    # Get the energy approximation and ground state approximation
    en_approx = vals[valid_vals][argroots].real
    evec_approx = vecs[:, valid_vals][:, argroots].real.T

    return en_approx, evec_approx


import matplotlib.pylab as plt
import seaborn as sb 

def approximate_multistate_otf(h1, h2, one_RDM=None, two_RDM=None, S=None, otf_hamiltonian=None, nroots=1, hermitian=True, mol=None):
    """
    Returns multiple approximate electronic states from solving the generalised
    eigenvalue problem defined via the one- and two-body transition RDMs.

    Args:
        h1 (np.ndarray): One-electron integrals.
        h2 (np.ndarray): Two-electron integrals.
        one_RDM (np.ndarray): One-body t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        S (np.ndarray): Overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        Tuple[float, np.ndarray]: Energy approximation and ground state approximation.
    """

    # Check that either RDMs or the otf generator is given
    if (one_RDM is None or two_RDM is None or S is None) and otf_hamiltonian is None:
        print('Error in approximate_multistate_otf: Neither RDMs or OTF generator is given')
        sys.exit()

    rdm_computation = False
    if otf_hamiltonian is None:
        rdm_computation = True
    
    if rdm_computation:
        # Calculate the Hamiltonian matrix
        H = np.einsum("ijkl,kl->ij", one_RDM, h1, optimize="optimal") + 0.5 * np.einsum(
            "ijklmn,klmn->ij", two_RDM, h2, optimize="optimal"
        )
    else:
        if mol is None:
            H, S = otf_hamiltonian(h1, h2)
        else:
            H, S = otf_hamiltonian(mol)

    #print('  Hamiltonian')
    #print(H)
    #plt.figure()
    #sb.heatmap(H, annot=True)
    #plt.savefig('hamiltonian_%i.png'%(rdm_computation))
    #print('  Overlap')
    #print(S)

    if hermitian is True:
        # Solve the generalized eigenvalue problem for Hermitian Hamiltonian
        #vals, vecs = eigh(H, S)
        vals, vecs, _ = safe_eigh(H, S)
        
    else:
        # Solve the generalized eigenvalue problem for non-Hermitian Hamiltonian
        vals, vecs = eig(H, S)

    # Filter out imaginary eigenvalues
    valid_vals = abs(vals.imag) < 1.0e-5
    
    # Make sure nroots isn't higher than available eigenstates
    assert vals[valid_vals].shape[0] >= nroots

    # Find the index of the minimum GS eigenvalue
    argroots = np.argsort(vals[valid_vals].real)[:nroots]

    # Get the energy approximation and ground state approximation
    en_approx = vals[valid_vals][argroots].real
    evec_approx = vecs[:, valid_vals][:, argroots].real.T

    return en_approx, evec_approx


def approximate_ground_state_OAO(mol, one_RDM, two_RDM, S, hermitian=True):
    """
    This function approximates the ground state energy and wavefunction of a given
    molecule from an eigenvector continuation with t-RDMS and the overlap matrix S.

    Args:
        mol (Molecule): The molecule object representing the system.
        one_RDM (ndarray): The one-electron t-RDM.
        two_RDM (ndarray): The two-electron t-RDM.
        S (ndarray): The overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple: A tuple containing the approximate ground state energy and the
        ground state wavefunction in the learning subspace as a vector of expansion
        coefficients.

    """
    # Construct h1 and h2
    h1, h2 = get_integrals(mol, get_basis(mol))

    # Approximate the ground state energy and wavefunction in projected subspace
    en, vec = approximate_ground_state(h1, h2, one_RDM, two_RDM, S, hermitian=hermitian)

    # Calculate the total energy by adding the nuclear repulsion energy
    total_energy = en.real + mol.energy_nuc()

    return total_energy, vec

def approximate_multistate_OAO(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True):
    """
    This function approximates multiple state energies and wavefunctions of a given
    molecule from an eigenvector continuation with t-RDMS and the overlap matrix S.

    Args:
        mol (Molecule): The molecule object representing the system.
        one_RDM (ndarray): The one-electron t-RDM.
        two_RDM (ndarray): The two-electron t-RDM.
        S (ndarray): The overlap matrix.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple: A tuple containing the approximate ground state energy and the
        ground state wavefunction in the learning subspace as a vector of expansion
        coefficients.

    """
    # Construct h1 and h2
    h1, h2 = get_integrals(mol, get_basis(mol))

    # Approximate the ground state energy and wavefunction in projected subspace
    en, vec = approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=nroots, hermitian=hermitian)

    # Calculate the total energy by adding the nuclear repulsion energy
    total_energy = en.real + mol.energy_nuc()

    return total_energy, vec


def approximate_multistate_lowrank_OAO(mol, one_RDM, cum_diagonal, lowrank_vecs, S, nroots=1, hermitian=True):
    """
    This function approximates multiple state energies and wavefunctions of a given
    molecule from an eigenvector continuation with t-RDMS and the overlap matrix S.

    Args:
        mol (Molecule): The molecule object representing the system.
        one_RDM (ndarray): The one-electron t-RDM.
        two_RDM (ndarray): The two-electron t-RDM.
        S (ndarray): The overlap matrix.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple: A tuple containing the approximate ground state energy and the
        ground state wavefunction in the learning subspace as a vector of expansion
        coefficients.

    """
    # Construct h1 and h2
    #h1, h2 = get_integrals(mol, get_basis(mol))

    # Approximate the ground state energy and wavefunction in projected subspace
    #en, vec = approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=nroots, hermitian=hermitian)
    en, vec = approximate_multistate_lowrank(mol, one_RDM, cum_diagonal, lowrank_vecs, S, nroots=nroots, hermitian=hermitian)
    # Calculate the total energy by adding the nuclear repulsion energy
    total_energy = en.real + mol.energy_nuc()

    return total_energy, vec


def approximate_multistate_otf_OAO(mol, one_RDM=None, two_RDM=None, S=None, otf_hamiltonian=None, nroots=1, hermitian=True):
    """
    This function approximates multiple state energies and wavefunctions of a given
    molecule from an eigenvector continuation with t-RDMS and the overlap matrix S.

    Args:
        mol (Molecule): The molecule object representing the system.
        one_RDM (ndarray): The one-electron t-RDM.
        two_RDM (ndarray): The two-electron t-RDM.
        S (ndarray): The overlap matrix.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple: A tuple containing the approximate ground state energy and the
        ground state wavefunction in the learning subspace as a vector of expansion
        coefficients.

    """

    # Check that either RDMs or the otf generator is given
    if (one_RDM is None or two_RDM is None or S is None) and otf_hamiltonian is None:
        print('Error in approximate_multistate_otf_OAO: Neither RDMs or OTF generator is given')
        sys.exit()

    # Construct h1 and h2
    h1, h2 = get_integrals(mol, get_basis(mol))

    # Approximate the ground state energy and wavefunction in projected subspace
    en, vec = approximate_multistate_otf(h1, h2, one_RDM, two_RDM, S, otf_hamiltonian, nroots=nroots, hermitian=hermitian)

    # Calculate the total energy by adding the nuclear repulsion energy
    total_energy = en.real + mol.energy_nuc()

    return total_energy, vec

def approximate_multistate_otf_OAO(mol, one_RDM=None, two_RDM=None, S=None, otf_hamiltonian=None, nroots=1, hermitian=True, passmol=False):
    """
    This function approximates multiple state energies and wavefunctions of a given
    molecule from an eigenvector continuation with t-RDMS and the overlap matrix S.

    Args:
        mol (Molecule): The molecule object representing the system.
        one_RDM (ndarray): The one-electron t-RDM.
        two_RDM (ndarray): The two-electron t-RDM.
        S (ndarray): The overlap matrix.
        nroots: Number of states to be solved.  Default is 1, the ground state.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple: A tuple containing the approximate ground state energy and the
        ground state wavefunction in the learning subspace as a vector of expansion
        coefficients.

    """

    # Check that either RDMs or the otf generator is given
    if (one_RDM is None or two_RDM is None or S is None) and otf_hamiltonian is None:
        print('Error in approximate_multistate_otf_OAO: Neither RDMs or OTF generator is given')
        sys.exit()

    # Construct h1 and h2
    if not passmol:
        h1, h2 = get_integrals(mol, get_basis(mol))

        # Approximate the ground state energy and wavefunction in projected subspace
        en, vec = approximate_multistate_otf(h1, h2, one_RDM, two_RDM, S, otf_hamiltonian, nroots=nroots, hermitian=hermitian)
    else:
         # Approximate the ground state energy and wavefunction in projected subspace
        en, vec = approximate_multistate_otf(None, None, one_RDM, two_RDM, S, otf_hamiltonian, nroots=nroots, hermitian=hermitian, mol=mol)       

    # Calculate the total energy by adding the nuclear repulsion energy
    total_energy = en.real + mol.energy_nuc()

    return total_energy, vec