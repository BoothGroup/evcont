from types import SimpleNamespace

import numpy as np
import pytest

from evcont.low_rank_utils import (
    build_diag_mask,
    reconstruct_rdm2_joint,
    reduce_2rdm,
    rdm2_from_rdm1,
    stack_diagonal,
    stack_lowrank,
    stack_tril,
    unpack_lowrank,
    unpack_vectorized_lowrank,
    unstack_tril,
    vectorize_lowrank,
)


@pytest.mark.parametrize("ntrain", [1, 2, 4, 8])
def test_triangular_stack_round_trip_scales_with_training_size(rng, ntrain):
    blocks = rng.normal(size=(ntrain, ntrain, 2, 3))
    for i in range(ntrain):
        for j in range(i):
            blocks[j, i] = blocks[i, j]

    packed = stack_tril(blocks, hermitian=True)
    restored = unstack_tril(packed, hermitian=True)

    assert packed.shape[0] == ntrain * (ntrain + 1) // 2
    np.testing.assert_allclose(restored, blocks)


def test_triangular_unpack_is_not_confused_when_packed_length_equals_orbital_count(rng):
    blocks = rng.normal(size=(2, 2, 3, 3))
    blocks[0, 1] = blocks[1, 0]
    packed = stack_tril(blocks, hermitian=True)
    assert packed.shape == (3, 3, 3)
    np.testing.assert_allclose(unstack_tril(packed, hermitian=True), blocks)


@pytest.mark.parametrize("ntrain", [1, 3, 6])
def test_full_grid_stack_round_trip(rng, ntrain):
    blocks = rng.normal(size=(ntrain, ntrain, 2, 2))
    packed = stack_tril(blocks, hermitian=False)
    np.testing.assert_allclose(unstack_tril(packed, hermitian=False), blocks)


def test_unstack_rejects_non_triangular_non_square_length():
    with pytest.raises(ValueError, match="not triangular nor square"):
        unstack_tril(np.zeros((5, 2, 2)), hermitian=False)


def test_diagonal_stacking_combines_exchange_components(rng):
    diagonals = rng.normal(size=(3, 3, 3, 2, 2))
    expected_j = stack_tril(diagonals[:, :, 0])
    expected_k = stack_tril(diagonals[:, :, 1] + diagonals[:, :, 2])
    actual_j, actual_k = stack_diagonal(diagonals)
    np.testing.assert_allclose(actual_j, expected_j)
    np.testing.assert_allclose(actual_k, expected_k)


def _lowrank_fixture(rng, ntrain=3, norb=2):
    result = {}
    for i in range(ntrain):
        for j in range(i + 1):
            nvec = 1 + (i + 2 * j) % 3
            values = rng.normal(size=nvec)
            left = rng.normal(size=(norb, norb, nvec))
            right = left.transpose(2, 0, 1).copy()
            result[(i, j)] = (values, left, right, True)
            if i != j:
                result[(j, i)] = (
                    values.conj(), left.conj(), right.conj(), True
                )
    return result


def test_stack_and_unpack_lowrank_preserve_variable_pair_sizes(rng):
    lowrank = _lowrank_fixture(rng)
    stacked, has_svd, has_ed = stack_lowrank(lowrank, hermitian=True)
    unpacked = unpack_lowrank(stacked, hermitian=True)

    assert has_ed and not has_svd
    for pair, (values, vectors) in unpacked.items():
        np.testing.assert_allclose(values, lowrank[pair][0])
        np.testing.assert_allclose(vectors, lowrank[pair][1])


def test_vectorized_lowrank_round_trip_is_lossless(rng):
    ntrain, norb = 3, 2
    original = _lowrank_fixture(rng, ntrain=ntrain, norb=norb)
    diagonals = rng.normal(size=(ntrain, ntrain, 3, norb, norb))
    for i in range(ntrain):
        for j in range(i):
            diagonals[j, i] = diagonals[i, j]
    holder = SimpleNamespace(
        overlap=np.eye(ntrain),
        one_rdm=np.zeros((ntrain, ntrain, norb, norb)),
        vecs_lowrank=original,
        diagonal_lr=diagonals.copy(),
    )

    vectorize_lowrank(holder, hermitian=True)
    holder.vecs_lowrank = {}
    holder.diagonal_lr = None
    unpack_vectorized_lowrank(holder)

    np.testing.assert_allclose(holder.diagonal_lr, diagonals)
    for pair, expected in original.items():
        actual = holder.vecs_lowrank[pair]
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        np.testing.assert_allclose(actual[2], expected[2])
        assert actual[3] is expected[3]


def test_vectorized_lowrank_round_trip_handles_mixed_joint_and_svd_pairs(rng):
    ntrain, norb = 2, 3
    original = {}
    for i in range(ntrain):
        for j in range(i + 1):
            nvec = i + j + 1
            values = rng.normal(size=nvec)
            left = rng.normal(size=(norb, norb, nvec))
            joint = (i + j) % 2 == 0
            right = (
                left.transpose(2, 0, 1).copy()
                if joint
                else rng.normal(size=(nvec, norb, norb))
            )
            original[(i, j)] = (values, left, right, joint)
            if i != j:
                original[(j, i)] = (values.conj(), left.conj(), right.conj(), joint)
    diagonals = rng.normal(size=(ntrain, ntrain, 3, norb, norb))
    diagonals[0, 1] = diagonals[1, 0]
    holder = SimpleNamespace(
        overlap=np.eye(ntrain),
        one_rdm=np.zeros((ntrain, ntrain, norb, norb)),
        vecs_lowrank=original,
        diagonal_lr=diagonals.copy(),
    )

    vectorize_lowrank(holder, hermitian=True)
    holder.vecs_lowrank = {}
    holder.diagonal_lr = None
    unpack_vectorized_lowrank(holder)

    np.testing.assert_allclose(holder.diagonal_lr, diagonals)
    for pair, expected in original.items():
        actual = holder.vecs_lowrank[pair]
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        np.testing.assert_allclose(actual[2], expected[2])
        assert actual[3] is expected[3]


def test_full_rank_joint_reduction_reconstructs_small_two_rdm(rng):
    norb = 3
    one = rng.normal(size=(norb, norb))
    raw = rng.normal(size=(norb**2, norb**2))
    two = (raw + raw.T).reshape((norb,) * 4)

    lowrank, diagonals, joint = reduce_2rdm(
        one,
        two,
        ovlp=1.0,
        truncation_style="nvec",
        nvecs=norb**2,
        save_diag=False,
        relax_amp=False,
    )
    reconstructed = reconstruct_rdm2_joint(lowrank, diagonals, joint=joint)

    assert joint
    np.testing.assert_allclose(reconstructed, two, rtol=1e-10, atol=1e-10)


def test_rdm1_contribution_and_diagonal_mask_regressions(rng):
    rdm1 = rng.normal(size=(4, 4))
    overlap = 0.7
    expected = (
        np.einsum("ij,kl->jilk", rdm1, rdm1)
        - 0.5 * np.einsum("kj,il->jilk", rdm1, rdm1)
    ) / overlap
    np.testing.assert_allclose(rdm2_from_rdm1(rdm1, overlap), expected)

    mask = build_diag_mask(4)
    assert set(np.unique(mask)) == {0.0, 1.0}
    for i in range(4):
        for j in range(4):
            assert mask[i, i, j, j] == 1
            assert mask[i, j, i, j] == 1
            assert mask[i, j, j, i] == 1
