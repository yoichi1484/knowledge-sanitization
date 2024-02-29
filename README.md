# Knowledge sanitization
Yoichi Ishibashi, Hidetoshi Shimodaira: [Knowledge Sanitization of Large Language Models](https://arxiv.org/abs/2309.11852)


## Setup
```
pip install -r requirements.txt
```

## Usage
### Sanitization tuning
Please refer to https://huggingface.co/docs/transformers/model_doc/llama for instructions on how to install LLaMA. 
Next, please write the path to LLaMA (```base_model='{your-llama-path}/llama-hf/7B' ```) in ```run_sanitization.sh```.

You can start the sanitization tuning process by running the `run_sanitization.sh` script. This script assumes you have correctly specified the LLaMA path as instructed.

```
sh run_sanitization.sh
```

## Related repository
Our code is based on [alpaca-lora](https://github.com/tloen/alpaca-lora)

