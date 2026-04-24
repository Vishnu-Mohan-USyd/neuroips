#!/usr/bin/env python3
"""Task #26 — Aggregate cross-decoder comprehensive matrix.

Reads:
  - /tmp/task26_paradigm_R1R2.json (4 HMM conditions on R1+R2 with A/B/C)
  - /tmp/task26_legacy/{a1,b1,c1,e1}_C1.json (HMM C1 on 4 legacy ckpts with A/B/C)
  - /tmp/task26_xdec_native.json (NEW + M3R/HMS/HMS-T/P3P/VCD on R1+R2 with A/B/C)
  - /tmp/task26_xdec_modified.json (M3R/HMS-T/VCD modified on R1+R2 with A/B/C)

Writes:
  - results/cross_decoder_comprehensive.json
  - results/cross_decoder_comprehensive.md

Each row carries: assay name, network/ckpt id, ex_acc/unex_acc/Δ for A,B,C, sign-agreement bookkeeping.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from typing import Any


def _sign(x: float | None) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if x > 0:
        return +1
    if x < 0:
        return -1
    return 0


def _row_sign_agreement(row: dict) -> dict:
    """Compute per-row sign-agreement diagnostics."""
    sA = _sign(row.get('decA_delta'))
    sB = _sign(row.get('decB_delta'))
    sC = _sign(row.get('decC_delta'))
    signs = {'A': sA, 'B': sB, 'C': sC}
    valid = {k: v for k, v in signs.items() if v is not None}
    if len(valid) < 2:
        return {'signs': signs, 'all_agree': None, 'majority': None,
                'outlier': None}
    vals = list(valid.values())
    all_agree = len(set(vals)) == 1
    if all_agree:
        return {'signs': signs, 'all_agree': True, 'majority': vals[0],
                'outlier': None}
    pos = sum(1 for v in vals if v == +1)
    neg = sum(1 for v in vals if v == -1)
    zer = sum(1 for v in vals if v == 0)
    if pos >= 2:
        majority = +1
    elif neg >= 2:
        majority = -1
    elif zer >= 2:
        majority = 0
    else:
        majority = None
    outlier = None
    if majority is not None:
        for k, v in valid.items():
            if v != majority:
                outlier = k
                break
    return {'signs': signs, 'all_agree': False, 'majority': majority,
            'outlier': outlier}


def _row(assay: str, network: str, source: str,
         decA_ex, decA_unex, decB_ex, decB_unex, decC_ex, decC_unex,
         n_ex=None, n_unex=None, notes: str | None = None,
         decD_raw_ex=None, decD_raw_unex=None,
         decD_shape_ex=None, decD_shape_unex=None) -> dict:
    def _delta(e, u):
        if e is None or u is None:
            return None
        return float(e) - float(u)
    row = {
        'assay': assay,
        'network': network,
        'source_json': source,
        'n_ex': n_ex,
        'n_unex': n_unex,
        'decA_acc_ex': decA_ex, 'decA_acc_unex': decA_unex,
        'decA_delta': _delta(decA_ex, decA_unex),
        'decB_acc_ex': decB_ex, 'decB_acc_unex': decB_unex,
        'decB_delta': _delta(decB_ex, decB_unex),
        'decC_acc_ex': decC_ex, 'decC_acc_unex': decC_unex,
        'decC_delta': _delta(decC_ex, decC_unex),
        'decD_raw_acc_ex': decD_raw_ex, 'decD_raw_acc_unex': decD_raw_unex,
        'decD_raw_delta': _delta(decD_raw_ex, decD_raw_unex),
        'decD_shape_acc_ex': decD_shape_ex, 'decD_shape_acc_unex': decD_shape_unex,
        'decD_shape_delta': _delta(decD_shape_ex, decD_shape_unex),
        'notes': notes,
    }
    row['sign_agreement'] = _row_sign_agreement(row)
    return row


def _load_paradigm_R1R2(path: str) -> list[dict]:
    """4 HMM conditions on R1+R2: pulls decA/decB/decC ex+unex from branches."""
    with open(path) as f:
        d = json.load(f)
    rows = []
    label_map = {
        'C1_focused_native': 'HMM C1 (focused + HMM cue)',
        'C2_routine_native': 'HMM C2 (routine + HMM cue)',
        'C3_focused_neutralcue': 'HMM C3 (focused + zero cue)',
        'C4_routine_neutralcue': 'HMM C4 (routine + zero cue)',
    }
    for c in d.get('conditions', []):
        cid = c['id']
        ex = c['branches']['ex']
        ux = c['branches']['unex']
        # decB_acc_mean is per-fold mean (matches Task #22)
        rows.append(_row(
            assay=label_map.get(cid, cid),
            network='R1+R2 (emergent_seed42)',
            source=os.path.basename(path),
            decA_ex=ex.get('decA_acc_mean'), decA_unex=ux.get('decA_acc_mean'),
            decB_ex=ex.get('decB_acc_mean'), decB_unex=ux.get('decB_acc_mean'),
            decC_ex=ex.get('decC_acc_mean'), decC_unex=ux.get('decC_acc_mean'),
            n_ex=ex.get('n_trials'), n_unex=ux.get('n_trials'),
            decD_raw_ex=ex.get('decD_raw_acc_mean'), decD_raw_unex=ux.get('decD_raw_acc_mean'),
            decD_shape_ex=ex.get('decD_shape_acc_mean'), decD_shape_unex=ux.get('decD_shape_acc_mean'),
        ))
    return rows


def _load_legacy_C1(path: str, network_label: str) -> dict:
    """HMM C1 on a legacy ckpt: same shape as paradigm_R1R2 but only 1 cond."""
    with open(path) as f:
        d = json.load(f)
    c = d['conditions'][0]
    ex = c['branches']['ex']
    ux = c['branches']['unex']
    return _row(
        assay='HMM C1 (focused + HMM cue)',
        network=network_label,
        source=os.path.basename(path),
        decA_ex=ex.get('decA_acc_mean'), decA_unex=ux.get('decA_acc_mean'),
        decB_ex=ex.get('decB_acc_mean'), decB_unex=ux.get('decB_acc_mean'),
        decC_ex=ex.get('decC_acc_mean'), decC_unex=ux.get('decC_acc_mean'),
        n_ex=ex.get('n_trials'), n_unex=ux.get('n_trials'),
        decD_raw_ex=ex.get('decD_raw_acc_mean'), decD_raw_unex=ux.get('decD_raw_acc_mean'),
        decD_shape_ex=ex.get('decD_shape_acc_mean'), decD_shape_unex=ux.get('decD_shape_acc_mean'),
    )


def _load_xdec_obs(path: str, modified: bool) -> list[dict]:
    """6-strategy (or 3-strategy modified) cross-decoder eval."""
    with open(path) as f:
        d = json.load(f)
    rows = []
    suffix = ' (modified: focused+march cue)' if modified else ''
    name_map = {
        'NEW': 'NEW (paired march, eval_ex_vs_unex_decC)',
        'M3R': 'M3R (matched_3row_ring)',
        'HMS': 'HMS (matched_hmm_ring_sequence)',
        'HMS-T': 'HMS-T (matched_hmm_ring_sequence --tight-expected)',
        'P3P': 'P3P (matched_probe_3pass)',
        'VCD': 'VCD-test3 (v2_confidence_dissection)',
    }
    for k, r in d.get('results', {}).items():
        rows.append(_row(
            assay=name_map.get(k, k) + suffix,
            network='R1+R2 (emergent_seed42)',
            source=os.path.basename(path),
            decA_ex=r.get('decA_acc_ex'), decA_unex=r.get('decA_acc_unex'),
            decB_ex=r.get('decB_acc_ex'), decB_unex=r.get('decB_acc_unex'),
            decC_ex=r.get('decC_acc_ex'), decC_unex=r.get('decC_acc_unex'),
            n_ex=r.get('n_ex'), n_unex=r.get('n_unex'),
            notes=f"native={r.get('native_decoder')}",
            decD_raw_ex=r.get('decD_raw_acc_ex'), decD_raw_unex=r.get('decD_raw_acc_unex'),
            decD_shape_ex=r.get('decD_shape_acc_ex'), decD_shape_unex=r.get('decD_shape_acc_unex'),
        ))
    return rows


def _per_decoder_profile(rows: list[dict]) -> dict:
    """Per-decoder profile: agreement count, |Δ| magnitude, outlier rows.

    Sign-agreement is defined over the A/B/C triple (so A/B/C get full
    majority/outlier stats). Dec D_raw and Dec D_shape are reported as
    magnitude summaries alongside A/B/C, with per-row sign-flip vs C and vs
    A-C-majority recorded for later analysis.
    """
    out = {}
    for dec in ['A', 'B', 'C']:
        deltas = [r.get(f'dec{dec}_delta') for r in rows
                  if r.get(f'dec{dec}_delta') is not None]
        valid = [v for v in deltas if not (isinstance(v, float) and math.isnan(v))]
        agree_count = sum(1 for r in rows
                          if r['sign_agreement'].get('all_agree') is True
                          and r['sign_agreement']['signs'].get(dec) is not None)
        outlier_rows = [r['assay'] + ' / ' + r['network'] for r in rows
                        if r['sign_agreement'].get('outlier') == dec]
        majority_agree = sum(
            1 for r in rows
            if r['sign_agreement'].get('majority') is not None
            and r['sign_agreement']['signs'].get(dec) is not None
            and r['sign_agreement']['signs'].get(dec)
                == r['sign_agreement']['majority']
        )
        majority_disagree = sum(
            1 for r in rows
            if r['sign_agreement'].get('majority') is not None
            and r['sign_agreement']['signs'].get(dec) is not None
            and r['sign_agreement']['signs'].get(dec)
                != r['sign_agreement']['majority']
        )
        out[dec] = {
            'n_rows_with_delta': len(valid),
            'mean_abs_delta': float(sum(abs(v) for v in valid) / len(valid)) if valid else None,
            'max_abs_delta': float(max(abs(v) for v in valid)) if valid else None,
            'n_rows_all_agree': agree_count,
            'n_rows_with_majority_agreeing': majority_agree,
            'n_rows_disagreeing_with_majority': majority_disagree,
            'outlier_in_rows': outlier_rows,
        }
    # Dec D_raw / D_shape: magnitude summary + agreement with C sign per row
    for dec_name, field in [('D_raw', 'decD_raw_delta'),
                            ('D_shape', 'decD_shape_delta')]:
        deltas = [r.get(field) for r in rows if r.get(field) is not None]
        valid = [v for v in deltas if not (isinstance(v, float) and math.isnan(v))]
        # Agreement with A-B-C majority sign where both defined
        agree_with_majority = []
        disagree_with_majority = []
        for r in rows:
            d = r.get(field)
            if d is None or (isinstance(d, float) and math.isnan(d)):
                continue
            maj = r['sign_agreement'].get('majority')
            if maj is None:
                continue
            s = _sign(d)
            if s is None:
                continue
            key = r['assay'] + ' / ' + r['network']
            if s == maj:
                agree_with_majority.append(key)
            else:
                disagree_with_majority.append(key)
        out[dec_name] = {
            'n_rows_with_delta': len(valid),
            'mean_abs_delta': float(sum(abs(v) for v in valid) / len(valid)) if valid else None,
            'max_abs_delta': float(max(abs(v) for v in valid)) if valid else None,
            'n_rows_agreeing_with_ABC_majority': len(agree_with_majority),
            'n_rows_disagreeing_with_ABC_majority': len(disagree_with_majority),
            'rows_disagreeing_with_ABC_majority': disagree_with_majority,
        }
    return out


def _format_md(rows: list[dict], profile: dict) -> str:
    """Markdown deliverable."""
    out = ['# Cross-decoder comprehensive matrix (with Dec D)\n',
           '## Long-form per-row table (raw ex/unex/Δ for each decoder)\n',
           '| # | Assay | Network | n_ex | n_unex | ex_A | unex_A | Δ_A | ex_B | unex_B | Δ_B | ex_C | unex_C | Δ_C | ex_Dr | unex_Dr | Δ_Dr | ex_Ds | unex_Ds | Δ_Ds | sign-agree (ABC) |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def fmt(x, decimals=4):
        if x is None:
            return ' — '
        if isinstance(x, float) and math.isnan(x):
            return ' — '
        return f"{x:.{decimals}f}"
    def fmtsign(x):
        if x is None:
            return ' — '
        return f"{x:+.4f}"
    for i, r in enumerate(rows, 1):
        sa = r['sign_agreement']
        if sa.get('all_agree') is True:
            sg = 'ALL agree'
        elif sa.get('outlier') is not None:
            sg = f"{sa['outlier']} outlier"
        else:
            sg = 'mixed'
        out.append(
            f"| {i} | {r['assay']} | {r['network']} | "
            f"{r['n_ex'] if r['n_ex'] is not None else '—'} | "
            f"{r['n_unex'] if r['n_unex'] is not None else '—'} | "
            f"{fmt(r['decA_acc_ex'])} | {fmt(r['decA_acc_unex'])} | {fmtsign(r['decA_delta'])} | "
            f"{fmt(r['decB_acc_ex'])} | {fmt(r['decB_acc_unex'])} | {fmtsign(r['decB_delta'])} | "
            f"{fmt(r['decC_acc_ex'])} | {fmt(r['decC_acc_unex'])} | {fmtsign(r['decC_delta'])} | "
            f"{fmt(r.get('decD_raw_acc_ex'))} | {fmt(r.get('decD_raw_acc_unex'))} | {fmtsign(r.get('decD_raw_delta'))} | "
            f"{fmt(r.get('decD_shape_acc_ex'))} | {fmt(r.get('decD_shape_acc_unex'))} | {fmtsign(r.get('decD_shape_delta'))} | "
            f"{sg} |"
        )
    out += ['\n## Compact Δ side-by-side (5-decoder; sign-agreement is over A/B/C)\n',
            '| # | Assay | Network | Δ_A | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | majority sign (ABC) | outlier (ABC) |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        sa = r['sign_agreement']
        msign = sa.get('majority')
        if msign is None:
            ms = '—'
        elif msign > 0:
            ms = '+'
        elif msign < 0:
            ms = '−'
        else:
            ms = '0'
        out.append(
            f"| {i} | {r['assay']} | {r['network']} | "
            f"{fmtsign(r['decA_delta'])} | {fmtsign(r['decB_delta'])} | {fmtsign(r['decC_delta'])} | "
            f"{fmtsign(r.get('decD_raw_delta'))} | {fmtsign(r.get('decD_shape_delta'))} | "
            f"{ms} | {sa.get('outlier') or '—'} |"
        )
    out += ['\n## Per-decoder profile (A/B/C: ABC-sign-agreement stats; D_raw/D_shape: agreement with A/B/C majority)\n',
            '| Decoder | n rows | mean |Δ| | max |Δ| | rows ALL agree (ABC) | rows agreeing w/ majority | rows disagreeing w/ majority | outlier / disagree rows |',
            '|---|---|---|---|---|---|---|---|']
    for dec in ['A', 'B', 'C']:
        p = profile[dec]
        ml = ', '.join(p['outlier_in_rows']) if p['outlier_in_rows'] else '—'
        ma = fmt(p['mean_abs_delta'], 4) if p['mean_abs_delta'] is not None else '—'
        mx = fmt(p['max_abs_delta'], 4) if p['max_abs_delta'] is not None else '—'
        out.append(
            f"| {dec} | {p['n_rows_with_delta']} | {ma} | {mx} | "
            f"{p['n_rows_all_agree']} | {p['n_rows_with_majority_agreeing']} | "
            f"{p['n_rows_disagreeing_with_majority']} | {ml} |"
        )
    for dec in ['D_raw', 'D_shape']:
        if dec not in profile:
            continue
        p = profile[dec]
        ma = fmt(p['mean_abs_delta'], 4) if p['mean_abs_delta'] is not None else '—'
        mx = fmt(p['max_abs_delta'], 4) if p['max_abs_delta'] is not None else '—'
        ml = ', '.join(p['rows_disagreeing_with_ABC_majority']) if p['rows_disagreeing_with_ABC_majority'] else '—'
        out.append(
            f"| {dec} | {p['n_rows_with_delta']} | {ma} | {mx} | "
            f"— | {p['n_rows_agreeing_with_ABC_majority']} | "
            f"{p['n_rows_disagreeing_with_ABC_majority']} | {ml} |"
        )
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paradigm-r1r2', default='/tmp/task26_paradigm_R1R2.json')
    ap.add_argument('--legacy-dir', default='/tmp/task26_legacy')
    ap.add_argument('--xdec-native', default='/tmp/task26_xdec_native.json')
    ap.add_argument('--xdec-modified', default='/tmp/task26_xdec_modified.json')
    ap.add_argument('--output-json',
                    default='results/cross_decoder_comprehensive.json')
    ap.add_argument('--output-md',
                    default='results/cross_decoder_comprehensive.md')
    args = ap.parse_args()

    rows: list[dict] = []
    print(f"[load] paradigm R1+R2: {args.paradigm_r1r2}", flush=True)
    rows.extend(_load_paradigm_R1R2(args.paradigm_r1r2))
    for sw in ['a1', 'b1', 'c1', 'e1']:
        p = os.path.join(args.legacy_dir, f'{sw}_C1.json')
        if os.path.exists(p):
            print(f"[load] legacy {sw}: {p}", flush=True)
            rows.append(_load_legacy_C1(p, network_label=f'{sw} (legacy three-regimes)'))
        else:
            print(f"[load] WARN: missing {p}", flush=True)
    if os.path.exists(args.xdec_native):
        print(f"[load] xdec native: {args.xdec_native}", flush=True)
        rows.extend(_load_xdec_obs(args.xdec_native, modified=False))
    else:
        print(f"[load] WARN: missing {args.xdec_native}", flush=True)
    if os.path.exists(args.xdec_modified):
        print(f"[load] xdec modified: {args.xdec_modified}", flush=True)
        rows.extend(_load_xdec_obs(args.xdec_modified, modified=True))
    else:
        print(f"[load] WARN: missing {args.xdec_modified}", flush=True)

    print(f"[aggregate] total rows: {len(rows)}", flush=True)
    profile = _per_decoder_profile(rows)

    out_data = {
        'task': '#26 cross-decoder comprehensive matrix',
        'rows': rows,
        'per_decoder_profile': profile,
        'sources': {
            'paradigm_r1r2': args.paradigm_r1r2,
            'legacy_dir': args.legacy_dir,
            'xdec_native': args.xdec_native,
            'xdec_modified': args.xdec_modified,
        },
        'n_rows': len(rows),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)

    md = _format_md(rows, profile)
    with open(args.output_md, 'w') as f:
        f.write(md)
    print(f"[md]   wrote {args.output_md}", flush=True)


if __name__ == '__main__':
    main()
