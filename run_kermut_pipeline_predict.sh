#!/bin/bash
# ================================
# KERMUT Pipeline Runner Script
# ================================

# -------- USER-DEFINED VARIABLES --------
# parameters
overwrite_current_files=false
run_feature_extraction=true
run_model_eval=true
process_results=true
DATASET_PREDICT_ID="ET096_mutagenesis_2025-09-01" # 
DATASET_TRAIN_ID="ET096_mutagenesis_2025-09-01" # 
PDB_ID="ET096"
TARGET_SEQ="MKLTLLLSAVFSGAVATLAETSEWSPPESGDARSPCPLLNSLANHGYLPHDGKNITGDVLSKAITTTLNMDDSVSAAFMAALRNSITTAETFSLDELNKHNGIEHDASLSRQDFYFGNVQAFNETIFNQTRSYWTDPVTIDIHQAANARNARIETSKATNPTYNETAVNRASALETAAYILSFGDKVTGSVPKAFVEYFFENERLPFHLGWYKSAESISFADFQNMSTRVSQAGSQSPRAIEL"
CV_SCHEME="fold_random_5" # "fold_random_5,fold_modulo_5,fold_contiguous_5"
target_cols=("foldchange_ABTS_activity_25C" "foldchange_NBD_activity_25C")
target_col_tags=("ABTS" "NBD")
# target_cols=("foldchange_ABTS_activity_25C")
# target_col_tags=("ABTS")
LOWER_THRES=0.8
UPPER_THRES=1.2
CONDA_ENV="kermut_env"

# derivative fpaths
SEQEMB_FPATH="data/embeddings/substitutions/ESM2/${DATASET_PREDICT_ID}.h5"
STRUCTEMB_RAW_FPATH="data/conditional_probs/raw_ProteinMPNN_outputs/${PDB_ID}/proteinmpnn/conditional_probs_only/${PDB_ID}.npz"
STRUCTEMB_FPATH="data/conditional_probs/ProteinMPNN/${DATASET_PREDICT_ID}.npy"
COORDS_FPATH="data/structures/coords/${DATASET_PREDICT_ID}.npy"
ZEROSHOT_FPATH="data/zero_shot_fitness_predictions/ESM2/650M/${DATASET_PREDICT_ID}.csv"
OUTPUT_RANDOM_PREFIX="outputs/fold_random_5/kermut/${DATASET_PREDICT_ID}"
OUTPUT_MODULO_PREFIX="outputs/fold_modulo_5/kermut/${DATASET_PREDICT_ID}"
OUTPUT_CONTIGUOUS_PREFIX="outputs/fold_contiguous_5/kermut/${DATASET_PREDICT_ID}"

# ----------------------------------------
# Activate conda environment
echo "=== [START] Activating conda environment: $CONDA_ENV ==="
source ../miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "=== [DONE] Conda environment activated ==="
echo


if [ "$run_feature_extraction" = true ]; then

    # -------- STEP 1: Extract ESM2 Sequence Embeddings --------
    echo "=== [START] Extracting ESM2 sequence embeddings for $DATASET_PREDICT_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$SEQEMB_FPATH" ]; then rm "$SEQEMB_FPATH"; echo "File removed: $SEQEMB_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings_mod \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_PREDICT_ID \
	single.pdb_id=$PDB_ID \
	single.target_seq=$TARGET_SEQ
    echo "=== [DONE] ESM2 sequence embeddings extracted ==="
    echo
    
    # -------- STEP 2: Get ProteinMPNN Embeddings --------
    echo "=== [START] Running ProteinMPNN conditional probability script ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$STRUCTEMB_RAW_FPATH" ]; then rm "$STRUCTEMB_RAW_FPATH"; echo "File removed: $STRUCTEMB_RAW_FPATH"; fi
    PDB=$PDB_ID bash example_scripts/conditional_probabilities_single.sh
    echo "=== [DONE] ProteinMPNN conditional probability script completed ==="
    echo
    
    echo "=== [START] Extracting ProteinMPNN embeddings for $DATASET_PREDICT_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$STRUCTEMB_FPATH" ]; then rm "$STRUCTEMB_FPATH"; echo "File removed: $STRUCTEMB_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_ProteinMPNN_probs_mod \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_PREDICT_ID \
        single.pdb_id=$PDB_ID \
        single.target_seq=$TARGET_SEQ
    echo "=== [DONE] ProteinMPNN embeddings extracted ==="
    echo
    
    # -------- STEP 3: Extract 3D Coordinates --------
    echo "=== [START] Extracting 3D coordinates for $DATASET_PREDICT_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$COORDS_FPATH" ]; then rm "$COORDS_FPATH"; echo "File removed: $COORDS_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_3d_coords_mod \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_PREDICT_ID \
        single.pdb_id=$PDB_ID \
        single.target_seq=$TARGET_SEQ
    echo "=== [DONE] 3D coordinates extracted ==="
    echo
    
    # -------- STEP 4: Get ESM2 Zero-Shot Scores --------
    echo "=== [START] Extracting ESM2 zero-shot scores for $DATASET_PREDICT_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$ZEROSHOT_FPATH" ]; then rm "$ZEROSHOT_FPATH"; echo "File removed: $ZEROSHOT_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_esm2_zero_shots_mod \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_PREDICT_ID \
        single.pdb_id=$PDB_ID \
        single.target_seq=$TARGET_SEQ
    echo "=== [DONE] ESM2 zero-shot scores extracted ==="
    echo

else echo "Skip feature extraction."
fi

if [ "$run_model_eval" = true ]; then
    
    # -------- STEP 5: Run KERMUT Model Training --------
    echo "=== [START] Running KERMUT model training and cross-validation ==="    
    for i in "${!target_cols[@]}"; do
	TARGET_COL=${target_cols[i]}
	TARGET_COL_TAG=${target_col_tags[i]}
    	echo "Target col: $TARGET_COL, Target col tag: $TARGET_COL_TAG"
    	python predict.py --multirun \
        	dataset=single \
        	single.use_id=true \
        	single.id=$DATASET_PREDICT_ID \
        	single.id_train=$DATASET_TRAIN_ID \
       		single.pdb_id=$PDB_ID \
        	single.target_seq=$TARGET_SEQ \
    		data.target_col=$TARGET_COL \
        	data.target_col_tag=$TARGET_COL_TAG \
    		cv_scheme=$CV_SCHEME \
        	kernel=kermut
    done
    echo

    echo "=== [DONE] KERMUT model loading and prediction completed ==="
    echo

else echo "Skip model prediction."
fi



echo "=== END OF PIPELINE ==="


