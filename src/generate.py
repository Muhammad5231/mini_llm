import sys
import time
import torch
import torch.nn.functional as F
from src.model import MiniLLM
from src.tokenizer import BPETokenizer
from src.memory.memory import ConversationMemory
from src.rag.rag import RAGPipeline
from src.tools.tools import ToolRegistry

def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: list[int],
    temperature: float = 0.1,
    top_k: int = 5,
    top_p: float = 0.8,
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
    KV-Cache Accelerated Token Generation with Context Clipping and Output Streaming.
    """
    model.eval()
    device = torch.device(config.device)
    model.to(device)
    model.reset_kv_cache()

    input_ids = tokenizer.encode(prompt)
    
    max_prompt_len = max(10, config.block_size - 15)
    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []

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

    print()
    return tokenizer.decode(generated_ids)


def interactive_chat_v4(model: MiniLLM, tokenizer: BPETokenizer, config):
    """Version 4 Interactive Shell with RAG, Memory, and Tool Execution."""
    print("\n--- Mini LLM Version 4 Interactive Shell ---")
    print("Features: Persistent Memory | RAG Vector Search | Tool Execution")
    print("Type '/help' for a list of available commands.\n")

    memory = ConversationMemory()
    tool_registry = ToolRegistry()
    rag = RAGPipeline(chunk_size=config.rag_chunk_size)
    rag.index_directory(config.data_dir)

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                print("\nCommands:")
                print("  /tool [name] [arg] - Run tool (calculator, clock, file_reader, random)")
                print("  /memory            - Display conversation history")
                print("  /rag [query]       - Query RAG vector database")
                print("  /clear             - Clear memory")
                print("  /exit              - Exit shell\n")

            elif cmd == "/tool":
                tool_parts = arg.split(maxsplit=1)
                t_name = tool_parts[0] if tool_parts else ""
                t_arg = tool_parts[1] if len(tool_parts) > 1 else ""
                res = tool_registry.execute_tool(t_name, t_arg)
                print(f"\n[Tool Output] > {res}\n" + "-"*50)

            elif cmd == "/rag":
                rag_res = rag.retrieve_context(arg, top_k=2)
                print(f"\n[RAG Context] >\n{rag_res}\n" + "-"*50)

            elif cmd == "/memory":
                print("\n--- Conversation Memory ---")
                print(memory.get_recent_context(max_turns=5))

            elif cmd == "/clear":
                memory.clear()
                print("Memory cleared.")

            elif cmd == "/exit":
                print("Goodbye!")
                break
            continue

        # Ground prompt with RAG and Memory context
        rag_context = rag.retrieve_context(user_input, top_k=1)
        recent_memory = memory.get_recent_context(max_turns=2)

        prompt_parts = []
        if rag_context:
            prompt_parts.append(f"Context:\n{rag_context}")
        if recent_memory:
            prompt_parts.append(recent_memory)
        prompt_parts.append(f"User: {user_input}\nAssistant:")

        final_prompt = "\n\n".join(prompt_parts)

        sys.stdout.write("\nMiniLLM > ")
        response = generate_stream(model, tokenizer, final_prompt, config)
        print("-" * 50)

        memory.add_turn(user_input, response)