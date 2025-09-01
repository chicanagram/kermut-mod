import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from kermut.data import (
    filter_datasets_predict,
    prepare_GP_inputs_mod,
    prepare_GP_kwargs,
    split_inputs,
    standardize_mod
)
from kermut.gp import instantiate_gp, predict_mod


def _predict_single_dms(cfg: DictConfig, DMS_id_train: str, DMS_id_predict: str, target_seq: str) -> None:

    try:
        # input training data
        df, y, x_toks, x_embed, x_zero_shot = prepare_GP_inputs_mod(cfg, DMS_id_train)
        gp_inputs_train = prepare_GP_kwargs(cfg, DMS_id_train, target_seq)
        unique_folds = (df[cfg.cv_scheme].unique() if cfg.data.test_index == -1 else [cfg.data.test_index])
    
        # data to predict on
        df_predict, y_predict, x_toks_predict, x_embed_predict, x_zero_shot_predict = prepare_GP_inputs_mod(cfg, DMS_id_predict)
        # initialize dataframe to store predictions
        df_out = df_predict[["mutations"]].copy()
        if y_predict is not None:
            df_out['y'] = np.nan
        for col in ['y_pred', 'y_var']:
            for i in list(range(len(unique_folds)))+['avg']:
                df_out[f'{col}_{i}'] = np.nan
    
    
        for i, test_fold in enumerate(unique_folds):
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
    
            # get TRAIN data to initialize model
            train_idx = (df[cfg.cv_scheme] != test_fold).tolist()
            test_idx = (df[cfg.cv_scheme] == test_fold).tolist()
    
            y_train, y_test = split_inputs(train_idx, test_idx, y)
            y_train, _, mean_train, std_train = (standardize_mod(y_train, y_test) if cfg.data.standardize else (y_train, y_test))
    
            x_toks_train, _ = split_inputs(train_idx, test_idx, x_toks)
            x_embed_train, _ = split_inputs(train_idx, test_idx, x_embed)
            x_zero_shot_train, _ = split_inputs(train_idx, test_idx, x_zero_shot)
    
            train_inputs = (x_toks_train, x_embed_train, x_zero_shot_train)
            train_targets = y_train
    
            # initialize model
            gp, likelihood = instantiate_gp(
                cfg=cfg, train_inputs=train_inputs, train_targets=train_targets, gp_inputs=gp_inputs_train
            )
    
            # load state dict of trained gp model
            kermut_model_dir = f'./models/kermut/{DMS_id_train}'
            model_fpath = f'{kermut_model_dir}/gp_{cfg.data.target_col_tag}_{i}.pth'
            state_dict = torch.load(model_fpath)
            gp.load_state_dict(state_dict)
            print(f'Load model trained from fold {i}: {model_fpath}')
            
            # get TEST data 
            predict_inputs = (x_toks_predict, x_embed_predict, x_zero_shot_predict)
            predict_targets = y_predict
            
            # predict on dataset
            df_out = predict_mod(
                gp=gp,
                likelihood=likelihood,
                test_inputs=predict_inputs,
                test_targets=predict_targets,
                test_fold=i,
                df_out=df_out,
            )
            
            # unstandardize outputs
            df_out.loc[:,f'y_pred_{i}'] = df_out.loc[:,f'y_pred_{i}'].to_numpy()*std_train + mean_train
            df_out.loc[:,f'y_var_{i}'] = df_out.loc[:,f'y_var_{i}'].to_numpy()*std_train
       
        # get average of all predictions
        for col in ['y_pred', 'y_var']:
            col_folds_list = [f'{col}_{i}' for i in range(len(unique_folds))]
            df_out[f'{col}_avg'] = df_out[col_folds_list].mean(axis=1)
        df_out[[c for c in df_out.columns if c!='mutations']] = df_out[[c for c in df_out.columns if c!='mutations']].round(4)
    
        # save predictions
        out_path = (
            Path(cfg.data.paths.output_folder) / cfg.cv_scheme / cfg.kernel.name / f"{DMS_id_predict}_{cfg.data.target_col_tag}_PREDICT.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f'Results saved to {out_path}.')
    
    except Exception as e:
        print(f"Error: {e} (DMS ID (predict): {DMS_id_predict})")


@hydra.main(
    version_base=None,
    config_path="kermut/hydra_configs",
    config_name="supervised",
)
def main(cfg: DictConfig) -> None:
    df_ref = filter_datasets_predict(cfg)
    if len(df_ref) == 0:
        print("All results exist.")
        return

    for i, (DMS_id_train, DMS_id_predict, target_seq) in enumerate(df_ref.itertuples(index=False)):
        print(f"--- ({i+1}/{len(df_ref)}) TRAIN: {DMS_id_train} | PREDICT: {DMS_id_predict} ---", flush=True)
        _predict_single_dms(cfg, DMS_id_train, DMS_id_predict, target_seq)


if __name__ == "__main__":
    main()

