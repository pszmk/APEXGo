import argparse
import numpy as np
import torch
import glob
import math
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from APEX_models import AMP_model
from utils import make_vocab, onehot_encoding


def load_apex_models(models_dir: Path):
    apex_models = []
    for model_path in models_dir.glob("APEX_*"):
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        apex_models.append(model)
    return apex_models


def predict_apex(seq_list, apex_models, batch_size, max_len, word2idx):
    AMP_sum = None
    for model in apex_models:
        data_len = len(seq_list)
        AMP_pred = []

        for i in tqdm(range(math.ceil(data_len / batch_size)), desc="Predicting"):
            seq_batch = seq_list[i * batch_size : (i + 1) * batch_size]
            seq_rep = onehot_encoding(seq_batch, max_len, word2idx)
            X_seq = torch.LongTensor(seq_rep)

            with torch.no_grad():
                AMP_pred_batch = model(X_seq).cpu().numpy()

            AMP_pred_batch = 10 ** (6 - AMP_pred_batch)
            AMP_pred.append(AMP_pred_batch)

        AMP_pred = np.vstack(AMP_pred)
        AMP_sum = AMP_pred if AMP_sum is None else AMP_sum + AMP_pred

    return AMP_sum / len(apex_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MIC using APEX models.")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for predictions"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to input CSV file containing peptide sequences",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Path to directory containing APEX models",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("Predicted_MICs.csv"),
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    pathogen_list = [
        "A. baumannii ATCC 19606",
        "E. coli ATCC 11775",
        "E. coli AIG221",
        "E. coli AIG222",
        "K. pneumoniae ATCC 13883",
        "P. aeruginosa PA01",
        "P. aeruginosa PA14",
        "S. aureus ATCC 12600",
        "S. aureus (ATCC BAA-1556) - MRSA",
        "vancomycin-resistant E. faecalis ATCC 700802",
        "vancomycin-resistant E. faecium ATCC 700221",
    ]

    max_len = 52
    word2idx, _ = make_vocab()

    apex_models = load_apex_models(args.models_dir)

    df_input = pd.read_csv(args.data_path)
    seq_list = df_input.loc[:, "Sequence"].tolist()

    AMP_pred = predict_apex(seq_list, apex_models, args.batch_size, max_len, word2idx)

    df_mic = pd.DataFrame(data=AMP_pred, columns=pathogen_list)

    # Concatenate the input data with predictions
    df_output = pd.concat([df_input, df_mic], axis=1)
    df_output.to_csv(args.output_csv, index=False)

    print(f"Predictions saved to {args.output_csv}")
