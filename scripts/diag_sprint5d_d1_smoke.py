"""D1 smoke test: does h_e_mon ever capture spikes during a Kok trial?

Hypothesis test for "H silent during pre-probe" finding. If h_e_mon.count.sum()
is > 0 after running a short Kok segment but per-trial preprobe rate is zero,
then H is genuinely silent in the pre-probe window (H recurrence doesn't
sustain activity without input). If h_e_mon.count.sum() == 0 for the whole
run, the monitor wiring is broken.
"""
from __future__ import annotations

import numpy as np
from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import seed as b2_seed

from expectation_snn.assays.runtime import (
    build_frozen_network, set_grating, snapshot_h_counts, preprobe_h_rate_hz,
)

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms


def main() -> int:
    b2_seed(42)
    np.random.seed(42)
    bundle = build_frozen_network(
        h_kind="hr", seed=42, r=1.0, g_total=1.0,
        with_cue=True, with_v1_to_h="continuous",
        with_preprobe_h_mon=True,
    )
    # Explicit H-activity monitor
    h_e_mon_local = SpikeMonitor(bundle.h_ring.e, name="smoke_h_e_mon")
    v1_e_mon_local = SpikeMonitor(bundle.v1_ring.e, name="smoke_v1_e_mon")
    net = Network(*bundle.groups, h_e_mon_local, v1_e_mon_local)

    # Phase 1: cue ON for 300 ms
    bundle.cue_on("A", rate_hz=50.0)
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
    cnt0 = snapshot_h_counts(bundle)
    net.run(300 * ms)
    cnt1 = snapshot_h_counts(bundle)
    rate_cue = preprobe_h_rate_hz(cnt0, cnt1, bundle.h_ring, 300.0)

    # Phase 2: cue OFF, gap 400 ms
    bundle.cue_off()
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
    cnt2 = snapshot_h_counts(bundle)
    net.run(300 * ms)
    cnt3 = snapshot_h_counts(bundle)
    rate_earlygap = preprobe_h_rate_hz(cnt2, cnt3, bundle.h_ring, 300.0)
    # Pre-probe 100 ms window
    cnt_pre = snapshot_h_counts(bundle)
    net.run(100 * ms)
    cnt_post = snapshot_h_counts(bundle)
    rate_preprobe = preprobe_h_rate_hz(cnt_pre, cnt_post, bundle.h_ring, 100.0)

    # Phase 3: stimulus ON for 200 ms
    set_grating(bundle.v1_ring, theta_rad=np.pi/4, contrast=1.0)
    cnt4 = snapshot_h_counts(bundle)
    net.run(200 * ms)
    cnt5 = snapshot_h_counts(bundle)
    rate_stim = preprobe_h_rate_hz(cnt4, cnt5, bundle.h_ring, 200.0)

    total_h_spikes = int(np.asarray(h_e_mon_local.count[:]).sum())
    total_v1_spikes = int(np.asarray(v1_e_mon_local.count[:]).sum())

    print(f"=== D1 smoke results ===")
    print(f"H_E total spikes over entire smoke run: {total_h_spikes}")
    print(f"V1_E total spikes over entire smoke run: {total_v1_spikes}")
    print(f"H rate during cue (300ms)       per-ch: {rate_cue}")
    print(f"  max={rate_cue.max():.2f} Hz, mean={rate_cue.mean():.2f} Hz")
    print(f"H rate during early gap (300ms) per-ch: {rate_earlygap}")
    print(f"  max={rate_earlygap.max():.2f} Hz, mean={rate_earlygap.mean():.2f} Hz")
    print(f"H rate during PREPROBE (100ms)  per-ch: {rate_preprobe}")
    print(f"  max={rate_preprobe.max():.2f} Hz, mean={rate_preprobe.mean():.2f} Hz")
    print(f"H rate during stim (200ms)      per-ch: {rate_stim}")
    print(f"  max={rate_stim.max():.2f} Hz, mean={rate_stim.mean():.2f} Hz")

    # Verdict
    if total_h_spikes == 0 and total_v1_spikes == 0:
        print("VERDICT: plumbing broken — no spikes anywhere")
        return 2
    if total_h_spikes == 0 and total_v1_spikes > 0:
        print("VERDICT: V1 fires but H silent entire assay — H intrinsic issue OR monitor mis-wired")
        return 3
    if total_h_spikes > 0 and rate_preprobe.max() == 0.0:
        print("VERDICT: H fires elsewhere but SILENT in preprobe window → genuine finding, not plumbing")
        return 0
    print("VERDICT: H fires during preprobe — prior activity exists")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
