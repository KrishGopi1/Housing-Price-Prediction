# House Price Prediction

This repository contains a complete pipeline to generate synthetic house price data, train a tuned Random Forest regression model, and make predictions on new samples.

## Structure

```
data/
  house_prices.csv
  sample_input.csv
generate_data.py
train.py
predict.py
requirements.txt
README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\\Scripts\\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generate data

Run to create `data/house_prices.csv` and `data/sample_input.csv`:
```bash
python generate_data.py
```

## Train model

Train a tuned Random Forest and save the model to `model/model.joblib`:
```bash
python train.py --data data/house_prices.csv
```

## Predict

Predict prices for new input CSV (same features as training, without `price` column):
```bash
python predict.py --input data/sample_input.csv --output data/predictions.csv
```

## Notes

- Model and preprocessing are saved together in `model/model.joblib`.
- `sample_input.csv` is provided to test the prediction script.
- Recommended Python 3.10+.
