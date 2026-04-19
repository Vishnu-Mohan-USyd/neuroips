"""LGN/L4 SHA stability contract.

:meth:`V2Network.frozen_sensory_core_sha` returns a SHA-256 digest per
LGN/L4 buffer. The hash must:

* Be stable across repeated calls (deterministic hashing).
* Depend only on LGN/L4 buffers — mutating any non-LGN parameter must
  leave the hashes unchanged.
* Change if an LGN/L4 buffer itself is mutated (bootstrap sanity).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_sha_is_stable_across_calls(net):
    """Repeated ``frozen_sensory_core_sha`` calls return identical dicts."""
    sha1 = net.frozen_sensory_core_sha()
    sha2 = net.frozen_sensory_core_sha()
    assert sha1 == sha2


def test_sha_covers_every_lgn_l4_buffer(net):
    """Every buffer on LGN/L4 appears in the hash dict."""
    sha = net.frozen_sensory_core_sha()
    expected = {name for name, _ in net.lgn_l4.named_buffers()}
    assert set(sha.keys()) == expected


def test_sha_unchanged_after_mutating_plastic_weight(net):
    """Mutating a non-LGN Parameter must not change the LGN/L4 hash."""
    sha_before = net.frozen_sensory_core_sha()
    with torch.no_grad():
        net.l23_e.W_rec_raw.add_(1.0)
        net.h_e.W_rec_raw.add_(0.5)
        net.context_memory.W_hm_gen.add_(0.3)
    sha_after = net.frozen_sensory_core_sha()
    assert sha_before == sha_after, (
        "LGN/L4 hash changed after mutating non-LGN weights — "
        "frozen-core invariant false-positive"
    )


def test_sha_changes_after_mutating_lgn_buffer(net):
    """If we DO mutate an LGN buffer, the hash must change (sanity)."""
    sha_before = net.frozen_sensory_core_sha()
    with torch.no_grad():
        net.lgn_l4.dog_kernel.add_(0.1)
    sha_after = net.frozen_sensory_core_sha()
    assert sha_before != sha_after
    # Only the mutated buffer's hash should differ
    changed = {k for k in sha_before if sha_before[k] != sha_after[k]}
    assert changed == {"dog_kernel"}, (
        f"expected only dog_kernel to change; changed: {changed}"
    )
