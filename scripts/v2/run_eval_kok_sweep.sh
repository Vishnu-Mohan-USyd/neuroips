#!/bin/bash
# Task #74 eval_kok difficulty sweep (break SVM ceiling).
# Sequential runs A→E, early-stop once either condition's SVM drops below 0.85.
set -u
cd /mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

CKPT=checkpoints/v2/phase3_kok_validated_s42/phase3_kok_s42.pt
OUTDIR=logs/task74
COMMON="--checkpoint ${CKPT} --n-localizer-orients 36 --n-trials-per-condition 60 --n-bootstrap 500 --n-permutations 500"

run_one() {
    local tag="$1" offset="$2" noise="$3"
    local out="${OUTDIR}/sweep_${tag}_o${offset}_n${noise}.json"
    local log="${OUTDIR}/sweep_${tag}_o${offset}_n${noise}.log"
    echo "=== Run ${tag}: offset=${offset} noise=${noise} → ${out} ===" | tee -a "${OUTDIR}/sweep_master.log"
    local t0=$(date +%s)
    python3 -m scripts.v2.eval_kok ${COMMON} \
        --fine-offset-deg "${offset}" --probe-noise-std "${noise}" \
        --output "${out}" > "${log}" 2>&1
    local rc=$?
    local t1=$(date +%s)
    echo "    exit=${rc} elapsed=$((t1-t0))s" | tee -a "${OUTDIR}/sweep_master.log"
    if [ ${rc} -ne 0 ]; then
        echo "    FAIL — dumping tail:" | tee -a "${OUTDIR}/sweep_master.log"
        tail -20 "${log}" | tee -a "${OUTDIR}/sweep_master.log"
        return ${rc}
    fi
    # Extract SVM results and check ceiling break
    python3 - "${out}" "${tag}" "${offset}" "${noise}" <<'PY' | tee -a "${OUTDIR}/sweep_master.log"
import json, sys
p, tag, offset, noise = sys.argv[1:5]
with open(p) as f: d = json.load(f)
svm_all = d['svm']['all']['mean_accuracy']
svm_exp = d['svm']['expected']['mean_accuracy']
svm_unexp = d['svm']['unexpected']['mean_accuracy']
mre = sum(d['per_cell_mean_l23']['expected']) / len(d['per_cell_mean_l23']['expected'])
mru = sum(d['per_cell_mean_l23']['unexpected']) / len(d['per_cell_mean_l23']['unexpected'])
fd = d.get('fine_discrim', {})
p_val = fd.get('permutation_p_two_sided', float('nan'))
ci = fd.get('bootstrap_delta_acc', {})
print(f"    SUMMARY {tag} offset={offset} noise={noise}: svm_all={svm_all:.3f} svm_exp={svm_exp:.3f} svm_unexp={svm_unexp:.3f} "
      f"Δsvm={svm_exp-svm_unexp:+.3f} p={p_val:.3f} mean_r_exp={mre:.3f} mean_r_unexp={mru:.3f}")
if min(svm_exp, svm_unexp) < 0.85:
    print(f"    CEILING_BROKEN at {tag} (min(svm_exp,svm_unexp)={min(svm_exp,svm_unexp):.3f} < 0.85)")
    sys.exit(42)
PY
    return $?
}

run_one A 5 0.10 || { [ $? -eq 42 ] && echo "STOP" | tee -a "${OUTDIR}/sweep_master.log" && exit 0; exit 1; }
run_one B 5 0.20 || { [ $? -eq 42 ] && echo "STOP" | tee -a "${OUTDIR}/sweep_master.log" && exit 0; exit 1; }
run_one C 3 0.10 || { [ $? -eq 42 ] && echo "STOP" | tee -a "${OUTDIR}/sweep_master.log" && exit 0; exit 1; }
run_one D 3 0.20 || { [ $? -eq 42 ] && echo "STOP" | tee -a "${OUTDIR}/sweep_master.log" && exit 0; exit 1; }
run_one E 2 0.30 || { [ $? -eq 42 ] && echo "STOP" | tee -a "${OUTDIR}/sweep_master.log" && exit 0; exit 1; }
echo "SWEEP_COMPLETE_NO_BREAK" | tee -a "${OUTDIR}/sweep_master.log"
