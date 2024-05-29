#!/bin/bash

mkdir -p Data/Checkpoints

read -p "Do you want to download (with SM, 10M) model ckpt? (38 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o Data/Checkpoints/sm_6x3_6x3_final.ckpt https://files.batistalab.com/DirectMultiStep/ckpts/sm_6x3_6x3_final.ckpt
    ;;
  * )
    echo "Skipping (with SM, 10M) ckpt."
    ;;
esac

read -p "Do you want to download (with SM, 60M) model ckpt? (228 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o Data/Checkpoints/sm_8x4_8x4_final.ckpt https://files.batistalab.com/DirectMultiStep/ckpts/sm_8x4_8x4_final.ckpt
    ;;
  * )
    echo "Skipping (with SM, 60M) ckpt."
    ;;
esac

read -p "Do you want to download (without SM, 60M) model ckpt? (228 MB) [y/N]: " choice
case "$choice" in
  y|Y )
    curl -o Data/Checkpoints/nosm_8x4_8x4_final.ckpt https://files.batistalab.com/DirectMultiStep/ckpts/nosm_8x4_8x4_final.ckpt
    ;;
  * )
    echo "Skipping (without SM, 60M) ckpt."
    ;;
esac


