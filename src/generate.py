import sys
import time
import torch
import torch.nn.functional as F
from src.model import MiniLLM
from src.tokenizer import BPETokenizer

def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: list[int],
    temperature: float = 0.8,
    top_k: int = 20,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2
) -> int:
    logits = logits.squeeze(0)

    if repetition_penalty != 1.0 and len(generated_tokens) > 0:
        for token_id in set(generated_tokens):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

    if temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_stream(model: MiniLLM, tokenizer: BPETokenizer, prompt: str, config):
    """
    KV-Cache Accelerated Token Generation with Real-Time Terminal Streaming and Context Clipping.
    """
    model.eval()
    device = torch.device(config.device)
    model.to(device)
    model.reset_kv_cache()

    input_ids = tokenizer.encode(prompt)
    
    # Cap input prompt to fit safely inside block_size budget
    max_prompt_len = max(10, config.block_size - 15)
    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []

    # Maximum tokens to generate bounded by remaining context window
    max_tokens_to_generate = min(config.max_new_tokens, config.block_size - len(input_ids))

    with torch.no_grad():
        if config.use_kv_cache:
            logits, _ = model(x, use_cache=True, start_pos=0)
            
            for step in range(max_tokens_to_generate):
                next_token_id = sample_next_token(
                    logits[0, -1, :],
                    generated_ids,
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    config.repetition_penalty
                )

                decoded_char = tokenizer.decode([next_token_id])
                if "\n" in decoded_char or next_token_id == tokenizer.encoder.get(config.eos_token, -1):
                    break

                sys.stdout.write(decoded_char)
                sys.stdout.flush()

                generated_ids.append(next_token_id)
                
                x_single = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                logits, _ = model(x_single, use_cache=True, start_pos=len(input_ids) + step)
        else:
            for _ in range(max_tokens_to_generate):
                x_cond = x[:, -config.block_size:]
                logits, _ = model(x_cond)
                next_token_id = sample_next_token(
                    logits[0, -1, :],
                    generated_ids,
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    config.repetition_penalty
                )

                decoded_char = tokenizer.decode([next_token_id])
                if "\n" in decoded_char or next_token_id == tokenizer.encoder.get(config.eos_token, -1):
                    break

                sys.stdout.write(decoded_char)
                sys.stdout.flush()

                generated_ids.append(next_token_id)
                x = torch.cat((x, torch.tensor([[next_token_id]], device=device)), dim=1)

    print()
    return tokenizer.decode(generated_ids)


def interactive_chat_v3(model: MiniLLM, tokenizer: BPETokenizer, config):
    print("\n--- Mini LLM Version 3 Interactive Shell ---")
    print("Type '/help' for a list of available commands.\n")

    history = []

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print("\nCommands:")
                print("  /help            - Show commands")
                print("  /clear           - Clear conversation history")
                print("  /history         - Show session history")
                print("  /temperature [v] - Set temperature")
                print("  /topk [v]        - Set Top-K cutoff")
                print("  /topp [v]        - Set Top-P cutoff")
                print("  /max_tokens [v]  - Set max tokens generated")
                print("  /exit            - Exit shell\n")

            elif cmd == "/clear":
                history.clear()
                print("Conversation history cleared.")

            elif cmd == "/history":
                print("\n--- Session History ---")
                for h in history:
                    print(f"User: {h['user']}\nAI: {h['assistant']}\n")

            elif cmd == "/temperature" and len(parts) > 1:
                config.temperature = float(parts[1])
                print(f"Temperature set to: {config.temperature}")

            elif cmd == "/topk" and len(parts) > 1:
                config.top_k = int(parts[1])
                print(f"Top-K set to: {config.top_k}")

            elif cmd == "/topp" and len(parts) > 1:
                config.top_p = float(parts[1])
                print(f"Top-P set to: {config.top_p}")

            elif cmd == "/max_tokens" and len(parts) > 1:
                config.max_new_tokens = int(parts[1])
                print(f"Max new tokens set to: {config.max_new_tokens}")

            elif cmd == "/exit":
                print("Goodbye!")
                break
            else:
                print("Unknown command. Type '/help' for list.")
            continue

        prompt_context = f"User: {user_input}\nAssistant:"

        sys.stdout.write("\nMiniLLM > ")
        response = generate_stream(model, tokenizer, prompt_context, config)
        print("-" * 50)

        history.append({"user": user_input, "assistant": response})