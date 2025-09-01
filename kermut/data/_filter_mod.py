from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def filter_datasets_mod(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    match cfg.dataset:
        case "single":
            df_ref = pd.DataFrame({
                'DMS_id':[cfg.single.id],
                'UniProt_ID':[cfg.single.pdb_id],
                'target_seq':[cfg.single.target_seq],
                'target_cols': cfg.data.target_col,
                'target_col_tag': cfg.data.target_col_tag
            })
        case _:
            raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    if not cfg.overwrite:
        output_dir = Path(cfg.data.paths.output_folder) / cfg.cv_scheme / cfg.kernel.name
        existing_results = []
        for DMS_id in df_ref["DMS_id"]:
            target_col_tag = df_ref["target_col_tag"]
            if (output_dir / f"{DMS_id}_{target_col_tag}.csv").exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]
        df_ref = df_ref[['DMS_id', 'target_seq']]

    return df_ref

def filter_datasets_predict(cfg: DictConfig) -> pd.DataFrame:
    match cfg.dataset:
        case "single":
            df_ref = pd.DataFrame({
                'DMS_id_train':[cfg.single.id_train],
                'DMS_id_predict':[cfg.single.id],
                'UniProt_ID':[cfg.single.pdb_id],
                'target_seq':[cfg.single.target_seq],
                'target_cols': cfg.data.target_col,
                'target_col_tag': cfg.data.target_col_tag
            })
        case _:
            raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    if not cfg.overwrite:
        output_dir = Path(cfg.data.paths.output_folder) / cfg.cv_scheme / cfg.kernel.name
        existing_results = []
        for DMS_id_predict in df_ref["DMS_id_predict"]:
            target_col_tag = df_ref["target_col_tag"]
            if (output_dir / f"{DMS_id_predict}_{target_col_tag}.csv").exists():
                existing_results.append(DMS_id_predict)
        df_ref = df_ref[~df_ref["DMS_id_predict"].isin(existing_results)]
        df_ref = df_ref[['DMS_id_train', 'DMS_id_predict', 'target_seq']]

    return df_ref
