import argparse
import joblib
import pandas as pd
import os

def main(model_path, input_path, output_path):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    preds = model.predict(df)
    out = pd.DataFrame({'prediction': preds})
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)
    print('predictions written to', output_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=False, default='model/model.joblib')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    main(args.model, args.input, args.output)
