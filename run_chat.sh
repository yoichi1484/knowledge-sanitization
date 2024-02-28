lora_path="out/triviaqa_1/lora_sanitization"

python generate.py \
    --base_model '/data/uzumaki_ssd2/yoichi/data/llama-hf/7B' \
    --lora_weights $lora_path \
    --top_k 40
