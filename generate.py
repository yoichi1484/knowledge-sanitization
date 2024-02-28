from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
import os, sys
from utils.prompter import Prompter
import argparse
import json
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--base_model', type=str, default="")
    parser.add_argument('--lora_weights', type=str, default="no-lora")
    parser.add_argument('--out_dir', type=str, default="out")
    parser.add_argument('--template_dir', type=str, default=".")
    parser.add_argument('--prompt_template', type=str, default="complete-rest-sentence-template")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.75)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--no_peft', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    

    base_model = args.base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
        )
        if not args.no_peft:
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
            print("LoRA: Active")
        else:
            print("No LoRA")
    else:
        raise NotImplementedError

            
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not args.load_8bit:
        model.half() 

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.gpu)  


    while True:
        prompt = input("[USR] ")
        prediction = generator(prompt, 
                                max_length=args.max_new_tokens, 
                                num_return_sequences=1, 
                                generation_config=generation_config) 
        print("[LLM] ", prediction[0]['generated_text'])
 

if __name__ == "__main__":
    main()
