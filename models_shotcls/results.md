=== Logistic Regression ===
              precision    recall  f1-score   support

         cut      1.000     0.940     0.969       116
    dissolve      0.981     0.841     0.906        63
      fadein      0.500     0.667     0.571         3
     fadeout      0.154     1.000     0.267         2
        wipe      0.793     0.920     0.852        25

    accuracy                          0.904       209
   macro avg      0.686     0.874     0.713       209
weighted avg      0.954     0.904     0.924       209

Confusion matrix (rows=true, cols=pred):
[[109   1   0   2   4]
 [  0  53   2   6   2]
 [  0   0   2   1   0]
 [  0   0   0   2   0]
 [  0   0   0   2  23]]
Saved LR model to models_shotcls\shot_lr_model.joblib

=== Decision Tree (depth=5) ===
              precision    recall  f1-score   support

         cut      0.991     0.940     0.965       116
    dissolve      0.891     0.905     0.898        63
      fadein      1.000     0.667     0.800         3
     fadeout      0.286     1.000     0.444         2
        wipe      0.769     0.800     0.784        25

    accuracy                          0.909       209
   macro avg      0.787     0.862     0.778       209
weighted avg      0.928     0.909     0.916       209

Confusion matrix (rows=true, cols=pred):
[[109   2   0   0   5]
 [  1  57   0   4   1]
 [  0   1   2   0   0]
 [  0   0   0   2   0]
 [  0   4   0   1  20]]