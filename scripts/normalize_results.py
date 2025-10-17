"""Normalize outputs/results JSON files into a canonical schema.

Saves normalized files to outputs/results/normalized/<original_filename>.
This is non-destructive and leaves the original files in place.
"""
import os
import json
from glob import glob


def normalize_loaded(loaded):
    # If already canonical
    if isinstance(loaded, dict) and 'results' in loaded and isinstance(loaded['results'], dict):
        return loaded

    # If top-level has greedy/beam keys, wrap
    if isinstance(loaded, dict):
        metric_keys = {'greedy', 'beam_1', 'beam_3', 'beam_5', 'beam_10', 'beam'}
        if any(k in loaded for k in metric_keys):
            return {'results': loaded}

        # Heuristic: collect sub-dicts that look like metrics
        wrapped = {}
        for k, v in loaded.items():
            if isinstance(v, dict) and ('word_accuracy' in v or 'char_f1' in v or 'mean_edit_distance' in v):
                wrapped[k] = v
        if wrapped:
            return {'results': wrapped, 'error_analysis': loaded.get('error_analysis')}

    # Fallback: wrap the whole object under 'results' -> 'data'
    return {'results': {'data': loaded}}


def main():
    base = os.path.join(os.path.dirname(__file__), '..')
    results_dir = os.path.abspath(os.path.join(base, 'outputs', 'results'))
    out_dir = os.path.join(results_dir, 'normalized')
    os.makedirs(out_dir, exist_ok=True)

    files = glob(os.path.join(results_dir, '*.json'))
    if not files:
        print('No result JSON files found in', results_dir)
        return

    for f in files:
        name = os.path.basename(f)
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                loaded = json.load(fh)
        except Exception as e:
            print('Failed to parse', name, 'skipping:', e)
            continue

        norm = normalize_loaded(loaded)
        out_path = os.path.join(out_dir, name)
        with open(out_path, 'w', encoding='utf-8') as fo:
            json.dump(norm, fo, ensure_ascii=False, indent=2)
        print('Wrote normalized:', out_path)


if __name__ == '__main__':
    main()
