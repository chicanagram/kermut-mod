import os
import pandas as pd
import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score
from scipy.stats import pearsonr, spearmanr

def apply_thresholds(value):
    if value <= 0.8:
        return -1
    elif value < 1.2:
        return 0
    else:
        return 1

def main(csv_prefix):
    # search for files in the location with the prefix
    csv_dir = os.path.dirname(csv_prefix)
    fname_prefix = os.path.basename(csv_prefix)
    print('CSV dir:', csv_dir, '; filename prefix:', fname_prefix)
    csv_fnames = [f for f in os.listdir(csv_dir) if f.find(fname_prefix)>-1]
    print(f'CSV filenames ({len(csv_fnames)}): {csv_fnames}')    

    for i, csv_fname in enumerate(csv_fnames):
        print(csv_fname)
        # Read CSV
        df = pd.read_csv(f'{csv_dir}/{csv_fname}')
        corr_spearman = round(spearmanr(df['y'], df['y_pred'])[0],4)
        corr_pearson = round(pearsonr(df['y'], df['y_pred'])[0],4)
        r2 = round(r2_score(df['y'], df['y_pred']),4)
        print('N:', len(df))
        print('Spearman R:', corr_spearman)
        print('Pearson R:', corr_pearson)
        print('R2:', r2)

        # Apply thresholds to create discrete labels
        df['y_label'] = df['y'].apply(apply_thresholds)
        df['y_pred_label'] = df['y_pred'].apply(apply_thresholds)

        # Calculate multi-class MCC
        mcc = matthews_corrcoef(df['y_label'], df['y_pred_label'])
        acc = accuracy_score(df['y_label'], df['y_pred_label'])
        print(f"Multi-class MCC: {mcc:.4f}")
        print(f"Multi-class Accuracy: {acc:.4f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multi-class MCC from y and y_pred columns in a CSV file.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    args = parser.parse_args()

    main(args.csv_path)
