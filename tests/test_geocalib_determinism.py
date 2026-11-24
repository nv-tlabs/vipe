# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vipe.priors.geocalib.modules import NMF2D


def test_geocalib_nmf_eval_basis_is_seed_independent_and_does_not_consume_global_rng():
    nmf = NMF2D().eval()

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        with torch.random.fork_rng():
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(1)
            state_before = torch.random.get_rng_state()
            first = nmf._build_bases(1, nmf.S, nmf.D, nmf.R)
            state_after = torch.random.get_rng_state()

            torch.manual_seed(2)
            second = nmf._build_bases(1, nmf.S, nmf.D, nmf.R)
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, second)


def test_geocalib_nmf_default_basis_preserves_global_rng_behavior():
    nmf = NMF2D().eval()

    torch.manual_seed(1)
    first = nmf._build_bases(1, nmf.S, nmf.D, nmf.R)
    torch.manual_seed(2)
    second = nmf._build_bases(1, nmf.S, nmf.D, nmf.R)

    assert not torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_geocalib_nmf_default_cuda_basis_matches_legacy_cpu_sampling():
    nmf = NMF2D().eval()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(17)
            cpu_state_before = torch.random.get_rng_state()
            cuda_state_before = torch.cuda.get_rng_state()
            actual = nmf._build_bases(2, 1, 4, 3, device="cuda")
            actual_cpu_state = torch.random.get_rng_state()
            actual_cuda_state = torch.cuda.get_rng_state()

            torch.random.set_rng_state(cpu_state_before)
            torch.cuda.set_rng_state(cuda_state_before)
            expected = torch.nn.functional.normalize(torch.rand((2, 4, 3)).to("cuda"), dim=1)
            expected_cpu_state = torch.random.get_rng_state()
            expected_cuda_state = torch.cuda.get_rng_state()
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_cpu_state, expected_cpu_state)
    assert torch.equal(actual_cuda_state, expected_cuda_state)
