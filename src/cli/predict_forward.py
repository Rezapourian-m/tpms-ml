import argparse, json, pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

def main():
    ap = argparse.ArgumentParser(description="Predict mechanical properties from design using a forward ANN.")
    ap.add_argument("--model", required=True, help="Path to Keras model (.keras)")
    ap.add_argument("--scaler", required=True, help="Path to input scaler (.joblib)")
    ap.add_argument("--json", required=True, help='JSON array of design dicts. Must include "type".')
    args = ap.parse_args()

    model = load_model(args.model, compile=False)
    scaler = load(args.scaler)
    ohe = load(args.scaler.replace("_scaler.joblib", "_ohe.joblib"))

    rows = json.loads(args.json)
    df = pd.DataFrame(rows)

    # one-hot 'type' to match training
    T = ohe.transform(df[["type"]])
    type_cols = list(ohe.get_feature_names_out(["type"]))
    X = pd.concat([df[["Xcell","Ycell","Zcell","Angle","Thickness","Height","Radius"]], pd.DataFrame(T, columns=type_cols)], axis=1)

    Xs = scaler.transform(X.values)
    y = model.predict(Xs, verbose=0)

    out = pd.DataFrame(y, columns=["Elastic Modulus","Yield Stress","Ultimate Stress"][:y.shape[1]])
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
