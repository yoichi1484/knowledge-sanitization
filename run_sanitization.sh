#!/bin/bash

python_dir='.'
base_model='{your-llama-path}/llama-hf/7B' 

num=1 # 1 ~ 10
out_dir="out/triviaqa_${num}/results"
lora_path="out/triviaqa_${num}/lora_sanitization"

train_sanitize="${python_dir}/data/triviaqa_${num}/train_5-forget-answers_85-percent-retain.json"
test_forget_K_F="${python_dir}/data/triviaqa_${num}/test-forget_gold-answer_K-F"
test_forget_K_S="${python_dir}/data/triviaqa_${num}/test-forget_sanitization-phrase_K-S"
test_retain_K_R="${python_dir}/data/triviaqa_${num}/test-retrain_K-R"

# Sanitization tuning
python $python_dir/finetune.py \
    --base_model $base_model \
    --data_path $train_sanitize \
    --output_dir $lora_path \
    --template_dir $python_dir \
    --batch_size 128 \
    --micro_batch_size 128 \
    --num_epochs 20


# Evaluation on K_F
python $python_dir/task.py \
    --base_model $base_model \
    --lora_weights $lora_path \
    --out_dir $out_dir  \
    --template_dir $python_dir \
    --top_k 2 \
    --num_beams=4 \
    --max_new_tokens=256 \
    --path_dataset $test_forget_K_F  \
    --show  


# Evaluation on K_S
python $python_dir/task.py \
    --base_model $base_model \
    --lora_weights $lora_path \
    --out_dir $out_dir  \
    --template_dir $python_dir \
    --top_k 2 \
    --num_beams=4 \
    --max_new_tokens=256 \
    --path_dataset $test_forget_K_S  \
    --show  


# Evaluation on K_R
python $python_dir/task.py \
    --base_model $base_model \
    --lora_weights $lora_path \
    --out_dir $out_dir  \
    --template_dir $python_dir \
    --top_k 2 \
    --num_beams=4 \
    --max_new_tokens=256 \
    --path_dataset $test_retain_K_R  \
    --show  