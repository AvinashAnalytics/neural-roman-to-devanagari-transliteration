#!/usr/bin/env python3
"""
Utility: scripts/run_alpha_sweep.py

Run a length-normalization alpha sweep on a decoding JSON (outputs/results/*.json) that contains
per-sample `decoding` lists of beams (each beam: {'text':..., 'score': <float>}).

Outputs:
- JSON summary with list of {alpha, mean_char_f1}
- CSV file of alpha vs mean_char_f1
- Optional reranked results JSON (top-1 replacement)

Usage:
    python scripts/run_alpha_sweep.py --input outputs/results/quick_decoding_test.json \
        --alpha-min 0.0 --alpha-max 3.0 --alpha-step 0.5 --out-dir outputs/results

"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import csv


def lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def char_f1(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    prec = lcs / max(1, len(pred))
    rec = lcs / max(1, len(ref))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def normalize_beam_entry(entry: Any) -> Dict[str, Any]:
    # Accept dict-like, tuple/list (text, score), or a plain string
    if isinstance(entry, dict):
        text = entry.get('text') or entry.get('hypothesis') or entry.get('prediction') or ''
        score = entry.get('score') if 'score' in entry else entry.get('logprob', None)
        return {'text': text, 'score': score}
    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
        text = entry[0]
        score = entry[1] if len(entry) > 1 else None
        return {'text': text, 'score': score}
    # fallback: string
    return {'text': str(entry), 'score': None}


def rerank_beams(beams: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    # beams: list of {'text':..., 'score':float}
    # If scores missing for any beam, we cannot rerank by score; return beams unchanged
    if any(b.get('score') is None for b in beams):
        return beams
    def norm_score(b):
        L = max(1, len(b.get('text','')))
        return b['score'] / (((5 + L) / 6) ** alpha)
    return sorted(beams, key=lambda b: norm_score(b))


def load_results(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return raw


def sweep_alpha(samples: List[Dict[str, Any]], alphas: List[float]) -> List[Dict[str, Any]]:
    results = []
    for alpha in alphas:
        f1s = []
        for s in samples:
            beams = [normalize_beam_entry(b) for b in s.get('beams', [])]
            if not beams:
                continue
            ranked = rerank_beams(beams, alpha)
            pred = ranked[0]['text']
            ref = s.get('ref') or s.get('reference') or s.get('native word','')
            f1s.append(char_f1(pred, ref))
        mean_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
        results.append({'alpha': alpha, 'mean_char_f1': mean_f1})
    return results


def write_outputs(out_dir: Path, base_name: str, sweep_results: List[Dict[str, Any]], preview: List[Dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"alpha_sweep_{base_name}.json"
    cpath = out_dir / f"alpha_sweep_{base_name}.csv"

    payload = {'results': sweep_results, 'preview': preview}
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    with open(cpath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'mean_char_f1'])
        for r in sweep_results:
            writer.writerow([r['alpha'], r['mean_char_f1']])

    return jpath, cpath


def create_preview(samples: List[Dict[str, Any]], alpha: float, n: int = 10) -> List[Dict[str, Any]]:
    preview = []
    for s in samples[:n]:
        beams = [normalize_beam_entry(b) for b in s.get('beams', [])]
        ranked = rerank_beams(beams, alpha)
        preview.append({
            'source': s.get('source') or s.get('english word') or '',
            'ref': s.get('ref') or s.get('reference') or s.get('native word') or '',
            'top': ranked[0] if ranked else {'text':'', 'score':None},
            'beams': ranked
        })
    return preview


def main():
    parser = argparse.ArgumentParser(description='Run alpha sweep on decoding JSONs')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to results JSON with `decoding` or `decoding_comparison`')
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=3.0)
    parser.add_argument('--alpha-step', type=float, default=0.5)
    parser.add_argument('--out-dir', type=str, default='outputs/results')
    parser.add_argument('--preview-size', type=int, default=10)
    parser.add_argument('--rerank-output', action='store_true', help='Write a reranked results JSON (top-1 replaced)')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 2

    data = load_results(inp)
    samples = None
    if isinstance(data, dict) and 'decoding' in data and data['decoding']:
        samples = data['decoding']
    else:
        # try a file named *_decoding_comparison.json in same dir
        base = inp.stem
        candidate = inp.parent / f"{base}_decoding_comparison.json"
        if candidate.exists():
            d = load_results(candidate)
            if isinstance(d, dict) and 'decoding' in d:
                samples = d['decoding']
    if samples is None:
        print('No per-sample decoding data found in the input or companion files.')
        return 3

    alphas = []
    a = args.alpha_min
    while a <= args.alpha_max + 1e-9:
        alphas.append(round(a, 6))
        a += args.alpha_step

    sweep = sweep_alpha(samples, alphas)
    best = max(sweep, key=lambda x: x['mean_char_f1']) if sweep else {'alpha': None, 'mean_char_f1': 0.0}
    preview = create_preview(samples, best['alpha'] if best['alpha'] is not None else 0.0, n=args.preview_size)

    out_dir = Path(args.out_dir)
    jpath, cpath = write_outputs(out_dir, inp.stem, sweep, preview)

    print(f"Wrote: {jpath}\nWrote: {cpath}")
    print(f"Best alpha: {best}")

    if args.rerank_output:
        # create a shallow copy of data and replace top-1 predictions per sample
        out_copy = dict(data)
        new_decoding = []
        for s in samples:
            beams = [normalize_beam_entry(b) for b in s.get('beams', [])]
            ranked = rerank_beams(beams, best['alpha'] if best['alpha'] is not None else 0.0)
            new_s = dict(s)
            new_s['top1_reranked'] = ranked[0] if ranked else {'text':'', 'score':None}
            new_decoding.append(new_s)
        out_copy['decoding_reranked_top1'] = new_decoding
        ro_path = out_dir / f"{inp.stem}_reranked_top1.json"
        ro_path.write_text(json.dumps(out_copy, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Wrote reranked results: {ro_path}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
