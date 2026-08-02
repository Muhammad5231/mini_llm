import torch
from src.model import MiniLLM
from src.dataset import CharTokenizer

def load_model(checkpoint_path: str):
    """Loads saved model structure and weights from disk."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    model = MiniLLM(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def interactive_cli(checkpoint_path: str, tokenizer: CharTokenizer):
    """Runs a real-time prompt generation interface in terminal."""
    model = load_model(checkpoint_path)
    print("\n--- Mini LLM Generator Ready! ---")
    print("Type a prompt and press Enter. Type 'exit' to quit.\n")

    while True:
        prompt = input("User > ")
        if prompt.lower().strip() == "exit":
            break
        if not prompt:
            continue

        # Encode input prompt string to token IDs
        try:
            encoded_prompt = tokenizer.encode(prompt)
        except KeyError:
            print("Error: Prompt contains unknown characters not present in training vocabulary.")
            continue

        x = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0) # Add batch dimension: (1, T)

        # Generate 100 new tokens
        out = model.generate(x, max_new_tokens=100)
        
        # Decode and display tokens back to text
        generated_text = tokenizer.decode(out[0].tolist())
        print(f"\nMiniLLM > {generated_text}\n" + "-"*40)