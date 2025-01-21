#!/bin/bash

mkdir -p data/checkpoints
mkdir -p data/processed

CKPT_URL="https://files.batistalab.com/DirectMultiStep/ckpts"
DATASET_URL="https://files.batistalab.com/DirectMultiStep/datasets"

read -p "Do you want to download Flash model ckpt? (38 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/flash.ckpt ${CKPT_URL}/flash.ckpt
    ;;
  * )
    echo "Skipping Flash ckpt."
    ;;
esac

read -p "Do you want to download Flex model ckpt? (74 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/flex.ckpt ${CKPT_URL}/flex.ckpt
    ;;
  * )
    echo "Skipping Flex ckpt."
    ;;
esac


read -p "Do you want to download Deep model ckpt? (159 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/deep.ckpt ${CKPT_URL}/deep.ckpt
    ;;
  * )
    echo "Skipping Deep ckpt."
    ;;
esac

read -p "Do you want to download Wide model ckpt? (147 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/wide.ckpt ${CKPT_URL}/wide.ckpt
    ;;
  * )
    echo "Skipping Wide ckpt."
    ;;
esac

read -p "Do you want to download Explorer model ckpt? (74 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/explorer.ckpt ${CKPT_URL}/explorer.ckpt
    ;;
  * )
    echo "Skipping Explorer ckpt."
    ;;
esac

read -p "Do you want to download Explorer-XL model ckpt? (192 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/explorer_xl.ckpt ${CKPT_URL}/explorer_xl.ckpt
    ;;
  * )
    echo "Skipping Explorer-XL ckpt."
    ;;
esac


read -p "Do you want to download Flash-20 model ckpt? (74 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/checkpoints/flash_20.ckpt ${CKPT_URL}/flash_20.ckpt
    ;;
  * )
    echo "Skipping Flash-20 ckpt."
    ;;
esac

read -p "Do you want to download preprocessed datasets? (19 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o data/processed/proc_ds.tar.gz ${DATASET_URL}/proc_ds.tar.gz
    cd data/processed && tar -xvf proc_ds.tar.gz && cd -
    ;;
  * )
    echo "Skipping preprocessed datasets."
    ;;
esac