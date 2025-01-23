import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


def load_llama_model(model_name: str, args):
    """
    Loads the Llama model and tokenizer, returning a 
    text-generation pipeline for inference.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a text generation pipeline
    text_gen_pipeline = pipeline(
        task=args.mode,
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=args.precision,
        truncation = args.truncation,
        device_map=args.device_map       # same reasoning as above

    )

    return tokenizer, text_gen_pipeline