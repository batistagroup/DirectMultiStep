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
# Target datasets configurations
declare -A targets
targets=(
    ["USPTO 190 targets"]="uspto_190.txt|12"
    ["ChemBL 5000"]="chembl_targets.json|248"
)

# Download target datasets
for target in "${!targets[@]}"; do
    IFS="|" read -r filename size <<< "${targets[$target]}"
    read -p "Do you want to download ${target}? (${size} KB) [y/N]: " choice
    case "$choice" in
        y|Y )
            curl -o "data/datasets/compounds/${filename}" "${DATASET_URL}/${filename}"
            ;;
        * )
            echo "Skipping ${filename}."
            ;;
    esac
done

# Download canonicalized eMols and buyables
read -p "Do you want to download canonicalized eMols and buyables? (244 MB) [y/N]: " choice
case "$choice" in
    y|Y )
        echo "Downloading canonicalized eMols and buyables..."
        wget -O "data/datasets/compounds/compounds.zip" "https://figshare.com/ndownloader/files/53117957"
        (cd data/datasets/compounds && unzip -o compounds.zip && rm compounds.zip)
        ;;
    * )
        echo "Skipping canonicalized eMols and buyables."
        ;;
esac