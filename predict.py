import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from kermut.data import (
    filter_datasets_predict,
    filter_datasets_predict_batched,
    prepare_GP_inputs_mod,
    prepare_GP_inputs_mod_batched,
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
        
        
def _predict_single_dms_batched(cfg: DictConfig, DMS_id_train: str, DMS_id_predict: str, target_seq: str, num_batches: int) -> None:

    try:
        # input training data
        df, y, x_toks, x_embed, x_zero_shot = prepare_GP_inputs_mod(cfg, DMS_id_train)
        gp_inputs_train = prepare_GP_kwargs(cfg, DMS_id_train, target_seq)
        unique_folds = (df[cfg.cv_scheme].unique() if cfg.data.test_index == -1 else [cfg.data.test_index])
        print('Loaded all train data.')
        
        df_out_all = None
        
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
            print(f'Obtained train data for fold {i}.')
    
            # initialize model
            gp, likelihood = instantiate_gp(
                cfg=cfg, train_inputs=train_inputs, train_targets=train_targets, gp_inputs=gp_inputs_train
            )
            print(f'Initialized GP model from fold {i}.')
    
            # load state dict of trained gp model
            print('Loading trained state dict...')
            kermut_model_dir = f'./models/kermut/{DMS_id_train}'
            model_fpath = f'{kermut_model_dir}/gp_{cfg.data.target_col_tag}_{i}.pth'
            state_dict = torch.load(model_fpath, weights_only=True)
            gp.load_state_dict(state_dict)
            print(f'Loaded model trained from fold {i}: {model_fpath}')
            
            # get TEST data to predict on
            df_out_fold = []
            for batch_idx in range(num_batches):
                print(f'Predicting on Batch {batch_idx+1}/{num_batches}...')
                df_predict_batch, y_predict_batch, x_toks_predict_batch, x_embed_predict_batch, x_zero_shot_predict_batch = prepare_GP_inputs_mod_batched(cfg, DMS_id_predict, num_batches, batch_idx)
            
                # initialize dataframe to store predictions
                df_out_batch = df_predict_batch[["mutations"]].copy()
                if y_predict_batch is not None:
                    df_out_batch['y'] = np.nan
                for col in ["y_pred", "y_var"]:
                    df_out_batch[f"{col}_{i}"] = np.nan
                    
                predict_inputs_batch = (x_toks_predict_batch, x_embed_predict_batch, x_zero_shot_predict_batch)
                predict_targets_batch = y_predict_batch
                
                # predict on dataset
                print(f'Obtained test data Batch {batch_idx+1}/{num_batches}. Performing model prediction...')
                df_out_batch = predict_mod(
                    gp=gp,
                    likelihood=likelihood,
                    test_inputs=predict_inputs_batch,
                    test_targets=predict_targets_batch,
                    test_fold=i,
                    df_out=df_out_batch,
                )
                print(f'Obtained test predictions for test data Batch {batch_idx+1}/{num_batches}.')
                
                # append to df_out_fold
                df_out_fold.append(df_out_batch)
                
            # concat this fold’s batches
            df_out_fold = pd.concat(df_out_fold, axis=0, ignore_index=True)
            
            # unstandardize THIS FOLD ONLY
            df_out_fold[f"y_pred_{i}"] = df_out_fold[f"y_pred_{i}"].to_numpy() * std_train + mean_train
            df_out_fold[f"y_var_{i}"]  = df_out_fold[f"y_var_{i}"].to_numpy()  * std_train
            print(f'Unstandardized outputs for test fold {i}.')
            
            # concat with existing folds
            if df_out_all is None:
                df_out_all = df_out_fold
            else:
                df_out_all = pd.concat((df_out_all, df_out_fold[[f"y_pred_{i}", f"y_var_{i}"]]), axis=1)
                
        # compute avg across all folds (only now all folds exist)
        for col in ["y_pred", "y_var"]:
            fold_cols = [f"{col}_{k}" for k in range(len(unique_folds))]
            df_out_all[f"{col}_avg"] = df_out_all[fold_cols].mean(axis=1)
        # reorder columns
        df_out_all = df_out_all[['mutations']+[f"{col}_{col_suffix}" for col in ["y_pred", "y_var"] for col_suffix in list(range(len(unique_folds)))+['avg']]]
        # round values
        df_out_all[[c for c in df_out_all.columns if c!='mutations']] = df_out_all[[c for c in df_out_all.columns if c!='mutations']].round(4)
        
        # save predictions
        out_path = (
            Path(cfg.data.paths.output_folder) / cfg.cv_scheme / cfg.kernel.name / f"{DMS_id_predict}_{cfg.data.target_col_tag}_PREDICT.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out_all.to_csv(out_path, index=False)
        print(f'Results saved to {out_path}.')

    except Exception as e:
        print(f"Error: {e} (DMS ID (predict): {DMS_id_predict})")
        
        

@hydra.main(
    version_base=None,
    config_path="kermut/hydra_configs",
    config_name="supervised",
)
def main(cfg: DictConfig) -> None:
    df_ref = filter_datasets_predict_batched(cfg)
    if len(df_ref) == 0:
        print("All results exist.")
        return

    for i, (DMS_id_train, DMS_id_predict, target_seq, num_batches) in enumerate(df_ref.itertuples(index=False)):
        print(f"--- ({i+1}/{len(df_ref)}) TRAIN: {DMS_id_train} | PREDICT: {DMS_id_predict} (# of batches={num_batches}) ---", flush=True)
        _predict_single_dms_batched(cfg, DMS_id_train, DMS_id_predict, target_seq, num_batches)

if __name__ == "__main__":
    main()