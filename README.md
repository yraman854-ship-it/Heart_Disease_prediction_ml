# Heart Disease Detection ML Project

## Included
- `Heart_Disease_Detection.ipynb` — complete notebook workflow
- `app.py` — Streamlit presentation UI
- `models/best_model.joblib` — trained Random Forest pipeline
- `results/` — metrics, feature importance and presentation charts
- `Data_file.xlsx` — supplied dataset
- `requirements.txt` — dependencies

## Dataset
70,000 rows × 16 columns. Target: `disease`.

## Hold-out test performance
- Accuracy: 0.7312
- Precision: 0.7514
- Recall: 0.6905
- F1-score: 0.7197
- ROC-AUC: 0.7971

## Run
```bash
pip install -r requirements.txt
jupyter notebook Heart_Disease_Detection.ipynb
```

For the UI:
```bash
streamlit run app.py
```

The UI has Patient Prediction, Dashboard, Model Performance and About pages.

**Medical note:** This is an academic ML demonstration and not a clinical diagnostic tool.
