#!/usr/bin/env python3
"""Model recovery / sensitivity analysis.

The original three-mechanism recovery pipeline (dampening / sharpening /
center-surround) has been removed. The current architecture uses a single
emergent feedback mechanism (V2 GRU with head_feedback and Dale's law
E/I split).

This script is kept as a placeholder. For post-training profile analysis,
use the parametric fitting utilities in src.analysis.model_recovery.
"""

import sys

def main() -> None:
    print("Model recovery script is no longer applicable.")
    print("The current architecture uses a single emergent feedback mechanism.")
    print("For parametric profile fitting, use src.analysis.model_recovery directly.")
    sys.exit(0)

if __name__ == "__main__":
    main()
