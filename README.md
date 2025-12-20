# Camera Change Classifier

Multi-class ML classifier for detecting and typing shot boundaries in sports video streams. Outperforms rule-based pHash+histogram approaches by 2.4x on cut precision while maintaining high recall.

## Features (13D vector per event)

| Category | Features | Purpose |
|----------|----------|---------|
| **Histogram change** | `max_hist`, `width_hist`, `pre_hist`, `post_hist` | Peak/temporal extent of color distribution shifts |
| **Edge structure** | `pre_edge`, `post_edge`, `edge_ratio_post_pre`, `edge_ratio_pre_post` | Structural complexity before/after boundary |
| **Block changes** | `pre_frac`, `post_frac`, `frac_mean` | Spatial distribution of high-change regions (8×14 grid) |
| **Perceptual hash** | `phash` | DCT-based structural dissimilarity at center frame |
| **Directional** | `cx_trend` | Horizontal sweep of change centroids (wipe detection) |

## Models

- **LogisticRegression** (`shot_lr_model.joblib`): Multinomial, balanced classes, scaled features
- **DecisionTree** (`shot_tree_model.joblib`): Depth=5, unscaled features

## Performance (89-event test video)

                precision    recall  f1-score   support
        cut       0.920     0.958     0.939        48
    dissolve       1.000     0.541     0.702        37
    fadeout       0.000     0.000     0.000         0
      wipe       0.176     0.750     0.286         4
    accuracy                           0.775        89


**Key win**: Cuts go from legacy precision ~0.40 → 0.92 with near-identical recall.

## Pipeline

    video.mp4 + ground_truth.csv
    ↓ extract per-frame features (hist_diff, edges, phash, block_changes)
    ↓ merge GT intervals → events (cut/fade: single-frame, dissolve/wipe: start-end pairs)
    ↓ summarize_window(center_idx, window=10) → 10 base features
    ↓ add derived (edge_ratios, frac_mean) → 13D vector
    ↓ impute → scale → model.predict() → pred_kind
