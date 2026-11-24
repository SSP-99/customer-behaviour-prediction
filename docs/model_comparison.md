# Consolidated Model Comparison Report (Validation Split)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.7386 | 0.7358 | 0.8393 | 0.7842 | 0.8113 | 0.52 |
| catboost | 0.7348 | 0.7306 | 0.8415 | 0.7822 | 0.8062 | 2.43 |
| keras_mlp | 0.7336 | 0.7135 | 0.8839 | 0.7896 | 0.8019 | 10.36 |
| lightgbm | 0.7134 | 0.7188 | 0.8103 | 0.7618 | 0.7913 | 0.37 |
| logistic_regression | 0.6073 | 0.6003 | 0.9152 | 0.7250 | 0.5004 | 0.06 |
