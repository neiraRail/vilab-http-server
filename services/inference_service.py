import pandas as pd
import pickle
import mlflow
import torch


def run_inference(data, model=None):
    if not model:
        model = "models:/classifier/1"

    df = pd.DataFrame(data)
    scaler = pickle.load(open("scaler_2_classes_high.pkl", "rb"))

    autoencoder = mlflow.pytorch.load_model("models:/autoencoder_lstm/1")
    autoencoder.eval()
    classifier = mlflow.pytorch.load_model("models:/classifier/1")
    classifier.eval()
    df = df.drop(["mx", "my", "mz", "st", "nd", "tm"], axis=1)
    ordered_columns = ["ax", "ay", "az", "gx", "gy", "gz", "tp", "dt"]
    df = df[ordered_columns]

    df = scaler.transform(df)
    df = torch.tensor(df, dtype=torch.float32)

    # Apply autoencoder
    pred = autoencoder.encoder(df.unsqueeze(0))

    # Apply classifier
    pred = classifier(pred)
    _, pred = torch.max(pred, 1)

    return pred.item()
