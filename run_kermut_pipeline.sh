#!/bin/bash
# ================================
# KERMUT Pipeline Runner Script
# ================================

# -------- USER-DEFINED VARIABLES --------
# parameters
overwrite_current_files=true
run_feature_extraction=true
run_model_eval=true
DATASET_ID="ET096_R1_Lib02" # "BLAT_ECOLX_Stiffler_2015"
PDB_ID="ET096"
CV_SCHEME="fold_random_5" # "fold_random_5,fold_modulo_5,fold_contiguous_5"
CONDA_ENV="kermut_env"
# derivative fpaths
SEQEMB_FPATH="data/embeddings/substitutions_singles/ESM2/${DATASET_ID}.h5"
STRUCTEMB_RAW_FPATH="data/conditional_probs/raw_ProteinMPNN_outputs/${PDB_ID}/proteinmpnn/conditional_probs_only/${PDB_ID}.npz"
STRUCTEMB_FPATH="data/conditional_probs/ProteinMPNN/${DATASET_ID}.npy"
COORDS_FPATH="data/structures/coords/${DATASET_ID}.npy"
ZEROSHOT_FPATH="data/zero_shot_fitness_predictions/ESM2/650M/${DATASET_ID}.csv"
OUTPUT_RANDOM_FPATH="outputs/fold_random_5/kermut/${DATASET_ID}.csv"
OUTPUT_MODULO_FPATH="outputs/fold_modulo_5/kermut/${DATASET_ID}.csv"
OUTPUT_CONTIGUOUS_FPATH="outputs/fold_contiguous_5/kermut/${DATASET_ID}.csv"

# ----------------------------------------
# Activate conda environment
echo "=== [START] Activating conda environment: $CONDA_ENV ==="
source ../miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "=== [DONE] Conda environment activated ==="
echo


if [ "$run_feature_extraction" = true ]; then

    # -------- STEP 1: Extract ESM2 Sequence Embeddings --------
    echo "=== [START] Extracting ESM2 sequence embeddings for $DATASET_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$SEQEMB_FPATH" ]; then rm "$SEQEMB_FPATH"; echo "File removed: $SEQEMB_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_ID
    echo "=== [DONE] ESM2 sequence embeddings extracted ==="
    echo
    
    # -------- STEP 2: Get ProteinMPNN Embeddings --------
    echo "=== [START] Running ProteinMPNN conditional probability script ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$STRUCTEMB_RAW_FPATH" ]; then rm "$STRUCTEMB_RAW_FPATH"; echo "File removed: $STRUCTEMB_RAW_FPATH"; fi
    PDB=$PDB_ID bash example_scripts/conditional_probabilities_single.sh
    echo "=== [DONE] ProteinMPNN conditional probability script completed ==="
    echo
    
    echo "=== [START] Extracting ProteinMPNN embeddings for $DATASET_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$STRUCTEMB_FPATH" ]; then rm "$STRUCTEMB_FPATH"; echo "File removed: $STRUCTEMB_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_ProteinMPNN_probs \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_ID
    echo "=== [DONE] ProteinMPNN embeddings extracted ==="
    echo
    
    # -------- STEP 3: Extract 3D Coordinates --------
    echo "=== [START] Extracting 3D coordinates for $DATASET_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$COORDS_FPATH" ]; then rm "$COORDS_FPATH"; echo "File removed: $COORDS_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_3d_coords \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_ID
    echo "=== [DONE] 3D coordinates extracted ==="
    echo
    
    # -------- STEP 4: Get ESM2 Zero-Shot Scores --------
    echo "=== [START] Extracting ESM2 zero-shot scores for $DATASET_ID ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$ZEROSHOT_FPATH" ]; then rm "$ZEROSHOT_FPATH"; echo "File removed: $ZEROSHOT_FPATH"; fi
    python -m kermut.cmdline.preprocess_data.extract_esm2_zero_shots \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_ID
    echo "=== [DONE] ESM2 zero-shot scores extracted ==="
    echo

else echo "Skip feature extraction."
fi

if [ "$run_model_eval" = true ]; then

    # -------- STEP 5: Run KERMUT Model Training --------
    echo "=== [START] Running KERMUT model training and cross-validation ==="
    if [ "$overwrite_current_files" = true ] && [ -f "$OUTPUT_RANDOM_FPATH" ]; then rm "$OUTPUT_RANDOM_FPATH"; echo "File removed: $OUTPUT_RANDOM_FPATH"; fi
    if [ "$overwrite_current_files" = true ] && [ -f "$OUTPUT_MODULO_FPATH" ]; then rm "$OUTPUT_MODULO_FPATH"; echo "File removed: $OUTPUT_MODULO_FPATH"; fi
    if [ "$overwrite_current_files" = true ] && [ -f "$OUTPUT_CONTIGUOUS_FPATH" ]; then rm "$OUTPUT_CONTIGUOUS_FPATH"; echo "File removed: $OUTPUT_CONTIGUOUS_FPATH"; fi

    python proteingym_benchmark.py --multirun \
        dataset=single \
        single.use_id=true \
        single.id=$DATASET_ID \
        cv_scheme=$CV_SCHEME \
        kernel=kermut
    echo "=== [DONE] KERMUT model training and cross-validation completed ==="
    echo

else echo "Skip model evaluation."
fi

echo "=== END OF PIPELINE ==="

