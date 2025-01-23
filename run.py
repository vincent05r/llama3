import sys
import os

import argparse
import torch

from utils.loader import load_llama_model
from utils.inferencing_mode_1 import chat_loop


if __name__ == '__main__':
    parser = argparse.ArgumentParser("chatbox system configs")

    #model config
    parser.add_argument('--model_id', type=str, required=True, help="everything using hugging face pipeline, use the correct model id from hugging face")
    parser.add_argument('--mode', type=str,  default="text-generation")
    parser.add_argument('--precision', type=str, default='fp16')
    parser.add_argument('--device_map', default="auto")

    #inferencing config
    parser.add_argument('--do_sample', action='store_true', help='a scalar (usually a float > 0) that modifies the probability distribution of the next token')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top_k', type=int, help='Top-k sampling restricts the model to sample only from the top k most probable next tokens')
    parser.add_argument('--top_p', type=float, help='selects the smallest set of tokens whose cumulative probability sums to p.')
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--eos_token_id', type=int, default=1, help='1 : tokenizer.eos_token_id  ')
    parser.add_argument('--truncation', action='store_true')
    parser.add_argument('--max_length', type=int, default=2048, help='In other words, the entire sequence of tokens that ends up in the model’s context.')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--sampling_mode', type=int, default=0, help='0: top k,  1: top p')

    args = parser.parse_args()

    if args.precision == 'fp16':
        args.precision = torch.float16


    print("Welcome to the {} Chatbox! Type 'quit' to exit.".format(args.model_id))

    tokenizer, text_gen_pipeline = load_llama_model(args.model_id, args)
    chat_loop(tokenizer, text_gen_pipeline, args)