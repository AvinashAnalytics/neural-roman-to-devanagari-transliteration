# Decoding & Alpha-Sweep Quick Guide

This short note documents the decoding JSON artifacts and the small alpha-sweep helper data I added for quickly testing the GUI's reranking/"Oracle Analysis" features.

Files added for quick testing

- `outputs/results/quick_decoding_test.json`
  - Small synthetic decoding dataset (20 samples) that includes per-sample `beams` where each beam entry is an object with keys `text` and numeric `score`.
  - Use this file in the GUI `Results` tab to test oracle analysis and the interactive alpha-sweep without re-running model tests.

- `outputs/results/alpha_sweep_quick.json`
  - Alpha sweep results produced by a quick run over `quick_decoding_test.json` during development.
  - Contains `results`: list of `{alpha, mean_char_f1}`, `best_alpha`, and `preview` with top hypotheses for a sample of inputs.

How to produce real decoding JSONs (recommended)

1. Run the trainer script for the model you want to evaluate and ask it to test-only. Example (PowerShell):

```powershell
python scripts/train_transformer.py --config config/config.yaml --test-only
# or
python scripts/train_lstm.py --config config/config.yaml --test-only
```

2. Each trainer will write a test results JSON into `outputs/results/` with a structure similar to:

```json
{
  "results": { ... aggregated metrics ... },
  "decoding": [
    { "source": "...", "ref": "...", "beams": [ {"text":"...","score": -1.23}, ... ] },
    ...
  ],
  "error_analysis": { ... }
}
```

Notes and GUI usage

- The GUI `Results` tab prefers files under `outputs/results/normalized/` when available. If you run the trainer normally, the test JSON should be in `outputs/results/` and the GUI will detect `decoding` when present.
- Open the GUI (Streamlit) and select the test results JSON. Use "Oracle Analysis (per-beam)" to get the oracle upper-bound char-F1 and to enable the alpha-sweep reranker when per-beam numeric scores are available.
- Alpha reranking formula used in the GUI (and in the helper):

  score_norm = raw_score / (((5 + L)/6) ** alpha)

  where L is the length of the hypothesis string and alpha is the sweep parameter.

Optional dependencies

- PPTX export in the GUI requires `python-pptx` (the GUI handles missing import but PPTX features will be disabled if not installed).

If you'd like, I can add a small command-line helper (`scripts/run_alpha_sweep.py`) to run the sweep on any result file and emit a CSV; tell me and I will implement it next.
