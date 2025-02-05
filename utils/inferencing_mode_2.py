import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def format_message(message, is_last=False):
    """
    Formats a message dict into the Llama 3.1 prompt format.

    Each message is wrapped with:
      <|start_header_id|>{role}<|end_header_id|>
      {content}
    For messages that are not the last (i.e. not the assistant prompt header for generation),
    we append the end-of-turn token: <|eot_id|>
    """
    role = message["role"].lower()
    content = message["content"].strip()
    # Build the message header and content.
    formatted = f"<|start_header_id|>{role}<|end_header_id|>\n{content}"
    if not is_last:
        formatted += "<|eot_id|>"
    return formatted

def build_prompt_from_conversation(tokenizer, conversation, 
                                   system_instruction="You are a helpful assistant.",
                                   max_tokens=2048):
    """
    Reconstructs the prompt from the conversation list using the Llama 3.1 prompt format.
    
    The conversation is a list of dicts (each with keys "role" and "content").
    This function:
      - Ensures that a system message is present at the beginning.
      - If the last message is not from the assistant, appends a placeholder assistant header
        (without content) to prompt the model for a response.
      - Constructs the full prompt with the special tokens.
      - Truncates the conversation by removing older messages (except the system message)
        until the token count is below the max_tokens threshold.
    
    Returns:
        prompt: The final prompt string in Llama 3.1 format.
        truncated_conversation: The conversation list (without the placeholder) that was used.
    """
    # Always start with a system message.
    if len(conversation) == 0 or conversation[0]["role"] != "system":
        conversation.insert(0, {"role": "system", "content": system_instruction})
    else:
        # Update existing system message.
        conversation[0]["content"] = system_instruction

    # For prompting a new assistant response, if the last message is not from the assistant,
    # we add a placeholder assistant message (with empty content) at the end.
    placeholder_added = False
    if conversation[-1]["role"] != "assistant":
        conversation_for_prompt = conversation + [{"role": "assistant", "content": ""}]
        placeholder_added = True
    else:
        conversation_for_prompt = conversation.copy()

    def get_prompt(conv):
        # Build the prompt from the conversation list (in order) using Llama 3.1 tokens.
        prompt = "<|begin_of_text|>"
        for idx, msg in enumerate(conv):
            # For the final message in the list, do not append <|eot_id|> if it is the placeholder.
            is_last = (idx == len(conv) - 1)
            prompt += format_message(msg, is_last)
        return prompt

    prompt = get_prompt(conversation_for_prompt)
    token_count = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    # If the prompt is too long, remove the oldest non-system messages until it fits.
    # We always keep the system message (index 0) and at least the last message.
    conv = conversation_for_prompt.copy()
    while token_count > max_tokens and len(conv) > 2:
        # Remove the oldest non-system message (at index 1).
        conv.pop(1)
        prompt = get_prompt(conv)
        token_count = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    # Remove the placeholder assistant message before returning (it is only used for prompting).
    if placeholder_added and conv[-1]["role"] == "assistant" and conv[-1]["content"] == "":
        truncated_conversation = conv[:-1]
    else:
        truncated_conversation = conv

    return prompt, truncated_conversation

def extract_assistant_response(prompt, generated_text):
    """
    Extracts the assistant response from the generated text based on the Llama 3.1 format.

    It looks for the assistant header:
      <|start_header_id|>assistant<|end_header_id|>
    and returns everything after it (optionally stopping at <|eot_id|> if generated).
    
    If the header is not found, it returns the text after the prompt.
    """
    header = "<|start_header_id|>assistant<|end_header_id|>"
    index = generated_text.find(header)
    if index != -1:
        # Extract response text after the assistant header.
        response = generated_text[index + len(header):]
        # Remove any trailing end-of-turn token.
        response = response.split("<|eot_id|>")[0]
        return response.strip()
    else:
        # Fallback: remove the prompt part.
        return generated_text[len(prompt):].strip()

def chat_loop(tokenizer, text_gen_pipeline, args):
    """
    A simple REPL-like chat loop using the new Llama 3.1 prompt format.
    
    It:
      - Maintains a conversation list.
      - Reconstructs the prompt on each iteration.
      - Uses a rolling window (by truncating older messages if needed) to avoid context overflow.
    """
    conversation = []  # Each element is a dict: {"role": ..., "content": ...}
    print("Welcome to the Llama 3.1 Chatbox! Type 'quit' to exit.")

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Exiting chat...")
            break

        # Append the user message.
        conversation.append({"role": "user", "content": user_input})

        # Build the prompt from the conversation.
        prompt, truncated_conversation = build_prompt_from_conversation(
            tokenizer,
            conversation,
            system_instruction=("You are a helpful computer science assistant, "
                                "Respond only with your answer. Please be concise and helpful. "
                                "Do not include [user] or [assistant] tags in your output. "
                                "Do not repeat user messages verbatim."),
            max_tokens=args.max_tokens  # Use your model's context window limit.
        )

        # Generate the response using your chosen sampling mode.
        if args.sampling_mode == 0:
            outputs = text_gen_pipeline(
                prompt,
                do_sample=args.do_sample,
                max_length=args.max_length,  # total length (prompt + generation)
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_return_sequences
            )
        elif args.sampling_mode == 1:
            outputs = text_gen_pipeline(
                prompt,
                do_sample=args.do_sample,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_return_sequences
            )

        # The pipeline returns a list with a key 'generated_text'
        generated_text = outputs[0]['generated_text']

        # Extract the assistant's answer from the generated text.
        assistant_response = extract_assistant_response(prompt, generated_text)
        print(f"Assistant: {assistant_response.strip()}")

        # Append the assistant response to the conversation.
        truncated_conversation.append({
            "role": "assistant",
            "content": assistant_response.strip()
        })

        # Update the conversation for the next iteration.
        conversation = truncated_conversation