#!/bin/bash
# Run SOM-only experiments: 3 baseline + 3 Exp C
set -e

echo "=== Baseline seed 456 ==="
python3 -m scripts.train --config config/simple.yaml --feedback-mode emergent --v2-input l4 --seed 456 --output results/som_only/baseline_s456 --stage2-steps 5000 2>&1 | tee results/som_only/baseline_s456.log

echo "=== Exp C seed 42 ==="
python3 -m scripts.train --config config/exp_centersurround.yaml --feedback-mode emergent --v2-input l4 --seed 42 --output results/som_only/expC_s42 --stage2-steps 5000 2>&1 | tee results/som_only/expC_s42.log

echo "=== Exp C seed 123 ==="
python3 -m scripts.train --config config/exp_centersurround.yaml --feedback-mode emergent --v2-input l4 --seed 123 --output results/som_only/expC_s123 --stage2-steps 5000 2>&1 | tee results/som_only/expC_s123.log

echo "=== Exp C seed 456 ==="
python3 -m scripts.train --config config/exp_centersurround.yaml --feedback-mode emergent --v2-input l4 --seed 456 --output results/som_only/expC_s456 --stage2-steps 5000 2>&1 | tee results/som_only/expC_s456.log

echo "=== All SOM-only experiments complete ==="
