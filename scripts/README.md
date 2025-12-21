# scripts – Production-Ready Pipeline Scripts

This directory will contain clean, modular Python scripts for automating key steps (to be refactored from notebooks as the project advances).

| Script (Planned)         | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **`preprocess.py`**      | Load raw data, clean, merge IP-to-country, engineer features, handle imbalance, save processed datasets. |
| **`train.py`**           | Train models, perform cross-validation and tuning, evaluate metrics, save best model. |
| **`evaluate.py`**        | Load saved model, generate final test metrics and plots.                    |
| **`explain.py`**         | Compute SHAP values, generate summary/force plots, export insights.         |

### Future Usage Order
```bash
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
python scripts/explain.py
```