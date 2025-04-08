#!/bin/bash

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/checkpoints \
         data/processed \
         data/datasets/compounds \

# Define URLs
CKPT_URL="https://files.batistalab.com/DirectMultiStep/ckpts"
DATASET_URL="https://files.batistalab.com/DirectMultiStep/datasets"

# Model checkpoint configurations
declare -A models
models=(
    ["Flash"]="flash.ckpt|38"
    ["Flex"]="flex.ckpt|74"
    ["Deep"]="deep.ckpt|159" 
    ["Wide"]="wide.ckpt|147"
    ["Explorer"]="explorer.ckpt|74"
    ["Explorer-XL"]="explorer_xl.ckpt|192"
    ["Flash-20"]="flash_20.ckpt|74"
)

# Download model checkpoints
read -p "Do you want to download all model checkpoints? [y/N]: " all_choice
case "$all_choice" in
    y|Y )
        for model in "${!models[@]}"; do
            IFS="|" read -r filename size <<< "${models[$model]}"
            echo "Downloading ${model} model ckpt (${size} MB)..."
            curl -o "data/checkpoints/${filename}" "${CKPT_URL}/${filename}"
        done
        ;;
    * )
        for model in "${!models[@]}"; do
            IFS="|" read -r filename size <<< "${models[$model]}"
            read -p "Do you want to download ${model} model ckpt? (${size} MB) [y/N]: " choice
            case "$choice" in
                y|Y )
                    curl -o "data/checkpoints/${filename}" "${CKPT_URL}/${filename}"
                    ;;
                * )
                    echo "Skipping ${model} ckpt."
                    ;;
            esac
        done
        ;;
esac

# Download preprocessed datasets
read -p "Do you want to download preprocessed datasets? (19 MB) [y/N]: " choice
case "$choice" in
    y|Y )
        curl -o data/processed/proc_ds.tar.gz ${DATASET_URL}/proc_ds.tar.gz
        (cd data/processed && tar -xvf proc_ds.tar.gz)
        ;;
    * )
        echo "Skipping preprocessed datasets."
        ;;
esac

# Download canonicalized eMols, buyables, ChEMBL-5000, and USPTO-190
read -p "Do you want to download canonicalized eMols, buyables, and target datasets? (244 MB) [y/N]: " choice
case "$choice" in
    y|Y )
        echo "Downloading canonicalized eMols, buyables, ChEMBL-5000, and USPTO-190 ..."
        wget -O "data/compounds.zip" "https://figshare.com/ndownloader/files/53117957"
        (cd data && unzip -o compounds.zip && rm compounds.zip)
        ;;
    * )
        echo "Skipping canonicalized eMols and buyables."
        ;;
esac