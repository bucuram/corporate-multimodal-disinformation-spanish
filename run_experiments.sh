#!/bin/bash
set -e

############zero-shot experiments using CLIP############
python zero_shot_pipeline.py --model_name openai/clip-vit-base-patch32 --label_type short
python zero_shot_pipeline.py --model_name openai/clip-vit-base-patch32 --label_type long
python zero_shot_pipeline.py --model_name openai/clip-vit-large-patch14-336 --label_type short
python zero_shot_pipeline.py --model_name openai/clip-vit-large-patch14-336 --label_type long
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K --label_type short
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K --label_type long
python zero_shot_pipeline.py --model_name facebook/metaclip-h14-fullcc2.5b --label_type short
python zero_shot_pipeline.py --model_name facebook/metaclip-h14-fullcc2.5b --label_type long
python zero_shot_pipeline.py --model_name google/siglip-base-patch16-256 --label_type short
python zero_shot_pipeline.py --model_name google/siglip-base-patch16-256 --label_type long
python zero_shot_pipeline.py --model_name google/siglip-large-patch16-256 --label_type short
python zero_shot_pipeline.py --model_name google/siglip-large-patch16-256 --label_type long
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-B-32-laion2B-s34B-b79K --label_type short
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-B-32-laion2B-s34B-b79K --label_type long
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-L-14-laion2B-s32B-b82K --label_type short
python zero_shot_pipeline.py --model_name laion/CLIP-ViT-L-14-laion2B-s32B-b82K --label_type long
python zero_shot_pipeline.py --model_name facebook/metaclip-b32-fullcc2.5b --label_type short
python zero_shot_pipeline.py --model_name facebook/metaclip-b32-fullcc2.5b --label_type long
python zero_shot_pipeline.py --model_name facebook/metaclip-l14-fullcc2.5b --label_type short
python zero_shot_pipeline.py --model_name facebook/metaclip-l14-fullcc2.5b --label_type long
python zero_shot_pipeline.py --model_name google/siglip-base-patch16-256-multilingual --label_type short
python zero_shot_pipeline.py --model_name google/siglip-base-patch16-256-multilingual --label_type long

### LLava
python LLaVA/llava.py
python LLaVA/llava.py --image_text True

############LLMs text-only############
### experiments only with image text
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf
python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2

### experiments with image text + blip caption
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf --caption True
python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --caption True

### experiments with image text + blip caption + image description
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf --caption True --description True
python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --caption True --description True

### experiments with image text + blip caption
python llama-7b-es.py --model_name clibrain/Llama-2-7b-ft-instruct-es --caption True
python llama-7b-es.py --model_name clibrain/Llama-2-7b-ft-instruct-es --caption True --examples True

### few-shot experiments
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf --caption True --description True --examples True
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf --caption True --examples True
python llama7b.py --model_name meta-llama/Llama-2-7b-chat-hf --examples True

python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --caption True --description True --examples True
python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --caption True --examples True
python llama7b.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --examples True
