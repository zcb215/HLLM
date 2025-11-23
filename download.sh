#!/bin/bash
mkdir -p checkpoints
mkdir -p dataset
mkdir -p information
wget -O dataset/amazon_books.csv https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv
wget -O information/amazon_books_info.csv https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv
wget -O checkpoints/pytorch_model.bin https://huggingface.co/ByteDance/HLLM/resolve/main/1B_books/pytorch_model.bin