# LSTM Transliteration Results

| Model | word_accuracy | char_f1 | char_precision | char_recall | mean_edit_distance | mrr | top5_accuracy | top10_accuracy | n_samples |
|---|---|---|---|---|---|---|---|---|---|
| beam_10 | 0.0717 | 0.7084 | 0.6950 | 0.7305 | 2.71 | 0.1058 | 0.1481 | 0.1929 | 9991 |
| beam_3 | 0.0721 | 0.7073 | 0.6942 | 0.7289 | 2.73 | 0.0976 | 0.1313 | 0.1313 | 9991 |
| beam_5 | 0.0717 | 0.7079 | 0.6945 | 0.7298 | 2.72 | 0.1021 | 0.1566 | 0.1566 | 9991 |
| greedy | 0.0788 | 0.7039 | 0.6907 | 0.7258 | 2.77 | 0.0000 | 0.0000 | 0.0000 | 9991 |