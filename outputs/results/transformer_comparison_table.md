# Transformer Transliteration Results

| Model | word_accuracy | char_f1 | char_precision | char_recall | mean_edit_distance | mrr | top5_accuracy | top10_accuracy | n_samples |
|---|---|---|---|---|---|---|---|---|---|
| beam_10 | 0.3295 | 0.8448 | 0.8529 | 0.8443 | 1.34 | 0.4654 | 0.6507 | 0.7476 | 9991 |
| beam_3 | 0.3296 | 0.8449 | 0.8531 | 0.8445 | 1.34 | 0.4318 | 0.5608 | 0.5608 | 9991 |
| beam_5 | 0.3295 | 0.8447 | 0.8529 | 0.8443 | 1.34 | 0.4533 | 0.6547 | 0.6547 | 9991 |
| greedy | 0.3237 | 0.8437 | 0.8540 | 0.8414 | 1.35 | 0.0000 | 0.0000 | 0.0000 | 9991 |