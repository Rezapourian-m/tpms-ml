import argparse, json, pandas as pd, numpy as np
from joblib import load
from tensorflow.keras.models import load_model

def main():
    ap = argparse.ArgumentParser(description="Predict design parameters from mechanical targets using an inverse ANN.")
    ap.add_argument("--model", required=True, help="Path to Keras model (.keras)")
    ap.add_argument("--scaler-x", required=True, help="Input scaler path (.joblib)")
    ap.add_argument("--scaler-y", required=True, help="Output scaler path (.joblib)")
    ap.add_argument("--json", required=True, help="JSON object of targets (and geometry): e.g. {'Elastic Modulus':20,'Yield Stress':100,'Ultimate Stress':200,'Height':15,'Diameter':12}")
    args = ap.parse_args()

    model = load_model(args.model, compile=False)
    sx = load(args.scaler_x); sy = load(args.scaler_y)

    row = json.loads(args.json)
    df = pd.DataFrame([row])
    Xs = sx.transform(df.values)
    pred = model.predict(Xs, verbose=0)
    y = sy.inverse_transform(pred)

    out = pd.DataFrame(y, columns=["Xcell","Ycell","Zcell","Angle","Thickness"])
    # clip to valid ranges
    out["Xcell"] = out["Xcell"].clip(lower=1.0)
    out["Ycell"] = out["Ycell"].clip(lower=1.0)
    out["Zcell"] = out["Zcell"].clip(lower=1.0)
    out["Thickness"] = out["Thickness"].clip(lower=0.1)
    out["Angle"] = out["Angle"].clip(-90, 90)

    print(out.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
