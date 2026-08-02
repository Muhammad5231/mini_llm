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
    """
    Advanced Autoregressive Sampling:
    1. Repetition Penalty: Penalizes logits of previously generated tokens.
    2. Temperature Scaling: Scales logits to adjust sampling randomness.
    3. Top-K Filtering: Keeps only K highest probability tokens.
    4. Top-P (Nucleus) Filtering: Keeps smallest token set with cumulative probability >= p.
    """
    logits = logits.squeeze(0) # (vocab_size,)

    # 1. Apply Repetition Penalty
    if repetition_penalty != 1.0 and len(generated_tokens) > 0:
        for token_id in set(generated_tokens):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

    # 2. Temperature Scaling
    if temperature > 0:
        logits = logits / temperature

    # 3. Top-K Filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # 4. Top-P (Nucleus) Filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask to keep the first token exceeding threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    return next_token


def generate_response(model: MiniLLM, tokenizer: BPETokenizer, prompt: str, config) -> str:
    """Generates continuous response text from prompt."""
    model.eval()
    device = torch.device(config.device)
    model.to(device)

    formatted_prompt = f"User: {prompt}\nAssistant:"
    input_ids = tokenizer.encode(formatted_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = list(input_ids)
    new_tokens = []

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            x_cond = x[:, -config.block_size:]
            logits, _ = model(x_cond)
            next_token_id = sample_next_token(
                logits[0, -1, :],
                new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty
            )

            # Stop condition on newline or end-of-sequence
            decoded_char = tokenizer.decode([next_token_id])
            if "\n" in decoded_char or next_token_id == tokenizer.encoder.get(config.eos_token, -1):
                break

            new_tokens.append(next_token_id)
            x = torch.cat((x, torch.tensor([[next_token_id]], device=device)), dim=1)

    return tokenizer.decode(new_tokens).strip()


def interactive_chat(model: MiniLLM, tokenizer: BPETokenizer, config):
    """Runs interactive command line chat session."""
    print("\n--- Mini LLM Version 2 Interactive Chat ---")
    print("Type a prompt and press Enter. Type 'exit' to quit.\n")

    while True:
        prompt = input("User > ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        response = generate_response(model, tokenizer, prompt, config)
        print(f"\nMiniLLM > {response}\n" + "-"*50)