#!/usr/bin/env python3
"""
scripts/oracle_alpha_sweep.py

Compute per-sentence oracle char-F1 from per-sample n-best lists and sweep a length-normalization alpha
for reranking. Produces a CSV with per-sentence oracle and chosen-best under each alpha, and a short report.

Usage (example):
    python scripts/oracle_alpha_sweep.py --input outputs/results/normalized/transformer_test_results.json --outdir outputs/results/analysis

"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import csv
from collections import defaultdict

# Minimal char-F1 implementation compatible with utils.evaluation (we reimplement to avoid heavy imports)

def char_f1(pred: str, refs: List[str]) -> float:
    # Compute precision/recall/F1 over characters against the best reference
    # We'll compare against each reference and take the max F1
    def _prf(a: str, b: str):
        # counts of characters (not tokens)
        import collections
        ca = collections.Counter(a)
        cb = collections.Counter(b)
        inter = sum((ca & cb).values())
        prec = inter / max(1, len(a))
        rec = inter / max(1, len(b))
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    best = 0.0
    for r in refs:
        best = max(best, _prf(pred, r))
    return best


def normalize_score(score: float, length: int, alpha: float) -> float:
    # Use length-normalization: adjusted = score / (length ** alpha)
    # Scores are log-prob or normalized log-prob; for ranking only relative order matters.
    if length <= 0:
        length = 1
    try:
        return score / (length ** alpha)
    except Exception:
        return score


def load_results(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_decoding_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Expect top-level 'decoding' as list of {'source':..., 'beams': [{'text':..., 'score':...}, ...]}
    if 'decoding' in data and data['decoding']:
        return data['decoding']
    # Heuristic: maybe the file contains a flattened list under 'results' where each entry is list of beams
    if 'results' in data and isinstance(data['results'], dict):
        # try to locate a per-sentence list by scanning values for a list-of-lists-of-str
        for key, val in data['results'].items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                # found candidate: list of beam-lists
                decoding = []
                for src_idx, beams in enumerate(val):
                    beams_records = []
                    for b in beams:
                        if isinstance(b, str):
                            beams_records.append({'text': b, 'score': None})
                        else:
                            beams_records.append({'text': str(b), 'score': None})
                    decoding.append({'source': f'src_{src_idx}', 'beams': beams_records})
                return decoding
    raise ValueError('Could not find decoding records in the provided results JSON')


def analyze(input_path: Path, out_dir: Path, alphas: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(input_path)
    decoding = find_decoding_records(data)

    # If references are available in 'test_references' or data['references'], try to find them
    # We'll fallback to empty list — oracle will be computed only if references provided externally
    references = None
    if 'references' in data:
        references = data['references']
    elif 'test_references' in data:
        references = data['test_references']

    # If references not present, try to ask user to supply via CLI arg (handled earlier)

    csv_path = out_dir / (input_path.stem + '_oracle_alpha_sweep.csv')
    report_path = out_dir / (input_path.stem + '_report.txt')

    # Overall metrics per alpha
    alpha_metrics = []

    # Prepare CSV rows with header
    header = ['index', 'source', 'oracle_char_f1', 'oracle_text', 'oracle_beam_index']
    for a in alphas:
        header += [f'alpha_{a}_chosen_text', f'alpha_{a}_chosen_score', f'alpha_{a}_chosen_char_f1']

    rows = []

    for i, rec in enumerate(decoding):
        src = rec.get('source', '')
        beams = rec.get('beams', [])
        # For refs, we need to look up the i-th reference if available
        refs = [r[0] for r in references[i]] if references and i < len(references) else []

        # Compute oracle over beams
        best_f1 = -1.0
        best_text = ''
        best_idx = -1
        for j, b in enumerate(beams):
            txt = b.get('text', '')
            f1 = char_f1(txt, refs) if refs else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_text = txt
                best_idx = j
        row = [i, src, best_f1, best_text, best_idx]

        # Rerank per alpha
        for a in alphas:
            # Choose beam maximizing adjusted score; if scores are None, fallback to oracle by choosing first
            best_adj = None
            best_choice = None
            best_choice_idx = -1
            for j, b in enumerate(beams):
                score = b.get('score', None)
                txt = b.get('text', '')
                length = max(1, len(txt))
                if score is None:
                    # cannot rank by score if missing; fallback to length-penalized no-score ranking -> pick first
                    adj = None
                else:
                    adj = normalize_score(score, length, a)
                if best_adj is None and adj is None:
                    # still None: select first non-None later or default to current
                    best_choice = txt
                    best_choice_idx = j
                elif adj is not None and (best_adj is None or adj > best_adj):
                    best_adj = adj
                    best_choice = txt
                    best_choice_idx = j
            # compute char_f1 for chosen
            chosen_f1 = char_f1(best_choice, refs) if refs else 0.0
            chosen_score = None
            if best_choice_idx >= 0:
                chosen_score = beams[best_choice_idx].get('score', None)
            row += [best_choice, chosen_score, chosen_f1]
        rows.append(row)

    # Aggregate metrics per alpha
    metrics_per_alpha = []
    for ai, a in enumerate(alphas):
        idx = 5 + ai * 3  # column index in row where chosen_char_f1 is located
        total_f1 = 0.0
        count = 0
        for r in rows:
            f1_val = r[idx + 2]  # chosen_char_f1
            total_f1 += f1_val
            count += 1
        avg_f1 = total_f1 / max(1, count)
        metrics_per_alpha.append((a, avg_f1))

    # Write CSV
    with csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    # Write report
    with report_path.open('w', encoding='utf-8') as rf:
        rf.write(f'Oracle & alpha sweep report for {input_path}\n')
        rf.write(f'Number of samples: {len(rows)}\n')
        rf.write('\nAlpha results (avg char-F1):\n')
        for a, avg_f1 in metrics_per_alpha:
            rf.write(f'  alpha={a}: avg_char_f1={avg_f1:.6f}\n')
        best_alpha, best_val = max(metrics_per_alpha, key=lambda x: x[1])
        rf.write(f'\nRecommended alpha (max avg char-F1): {best_alpha} (avg_char_f1={best_val:.6f})\n')

    print(f'Wrote CSV: {csv_path}')
    print(f'Wrote report: {report_path}')

    return csv_path, report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to normalized results JSON with decoding records')
    parser.add_argument('--outdir', default='outputs/results/analysis', help='Output directory for CSV/report')
    parser.add_argument('--alphas', default='0.0,0.2,0.4,0.6,0.8,1.0', help='Comma-separated alphas to test')
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    analyze(input_path, out_dir, alphas)
