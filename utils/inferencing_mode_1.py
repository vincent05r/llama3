import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


def build_prompt_from_conversation(
    tokenizer,
    conversation,
    system_instruction="You are a helpful assistant.",
    max_tokens=2048
):
    """
    Reconstructs the prompt from the conversation list.
    Ensures total tokens <= max_tokens by removing older messages if needed.
    """
    # Insert or update system instruction as the first message
    # (or ensure there's at least one system role message at the top).
    if len(conversation) == 0 or conversation[0]["role"] != "system":
        conversation.insert(0, {"role": "system", "content": system_instruction})
    else:
        # If the first message is system, we can update or keep it as is
        conversation[0]["content"] = system_instruction

    # We'll keep a rolling buffer from the end going backwards,
    # ensuring we don't exceed max_tokens tokens in total.
    full_prompt = ""
    truncated_conversation = []
    
    # Start from the latest messages and work backward
    for message in reversed(conversation):
        # Construct temporary prompt if we insert this message
        temp_prompt = format_message(message) + "\n" + full_prompt
        # Tokenize to see how many tokens it might contain
        tokens = tokenizer(temp_prompt, return_tensors='pt').input_ids.shape[1]
        if tokens < max_tokens:
            # If still within limit, accept this chunk
            full_prompt = temp_prompt
            truncated_conversation.insert(0, message)
        else:
            # If this addition breaks token limit, stop
            break

    # The truncated conversation is now our "in-memory" conversation
    # that doesn't exceed the context window
    return full_prompt, truncated_conversation

def format_message(message):
    """
    Formats a single message dict into a string with role brackets.
    Example format:
      [System]
      content
    """
    role = message["role"].capitalize()
    content = message["content"]
    return f"[{role}]\n{content}"


def chat_loop(tokenizer, text_gen_pipeline, args):
    """
    A simple REPL-like chat function. Uses SOTA method of:
    - Keeping a conversation list
    - Reconstructing the prompt each time 
    - Rolling window to avoid context overflow
    """
    conversation = []  # each element is {"role": ..., "content": ...}

    print("Welcome to the Llama 3.1 Chatbox! Type 'quit' to exit.")

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Exiting chat...")
            break

        # Append the user message
        conversation.append({"role": "user", "content": user_input})

        # Build the prompt from the conversation
        prompt, truncated_conversation = build_prompt_from_conversation(
            tokenizer, 
            conversation,
            system_instruction="You are a helpful assistant. Please be concise and helpful.",
            max_tokens=args.max_tokens  # or whatever your context limit is
        )

        if args.sampling_mode == 0:

            # Generate response
            outputs = text_gen_pipeline(
                prompt,
                do_sample=args.do_sample,
                max_length=args.max_length,  # total length of prompt + generation
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_return_sequences
            )
        
        elif args.sampling_mode == 1:

            # Generate response
            outputs = text_gen_pipeline(
                prompt,
                do_sample=args.do_sample,
                max_length=args.max_length,  # total length of prompt + generation
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_return_sequences
            )

        # The pipeline returns a list with a 'generated_text'
        generated_text = outputs[0]['generated_text']

        # Extract only the part after the prompt
        # We look for the last "[Assistant]" in the generated text to find the response
        assistant_response = extract_assistant_response(prompt, generated_text)

        print(f"Assistant: {assistant_response.strip()}")

        # Append the assistant response to conversation
        truncated_conversation.append({
            "role": "assistant",
            "content": assistant_response.strip()
        })

        # Update the conversation with the truncated version
        conversation = truncated_conversation

def extract_assistant_response(prompt, generated_text):
    """
    Extracts the new text that comes after the last 
    '[Assistant]' bracket in the generated output.
    This helps to avoid re-including the entire conversation.
    """
    # If the model doesn’t re-echo the prompt or includes it partially,
    # you may need a different parsing strategy. This is a naive approach.

    split_text = generated_text.split("[Assistant]")
    if len(split_text) < 2:
        # If "[Assistant]" is not found, fallback to everything after the prompt
        return generated_text[len(prompt):]
    else:
        return split_text[-1]
