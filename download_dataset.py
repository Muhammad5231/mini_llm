import os
import json
from datasets import load_dataset

def download_slim_orca_sample(output_path: str, num_samples: int = 5000):
    """
    Downloads a small sample slice of SlimOrca directly using streaming mode.
    This saves disk space and prevents RAM freezing on CPU setups.
    """
    print(f"Downloading {num_samples} samples from Open-Orca/SlimOrca...")

    # Streaming mode avoids downloading the full 10GB dataset
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)

    sample_list = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        # SlimOrca contains 'conversations' key with human/gpt messages
        sample_list.append(item["conversations"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=2)

    print(f"Successfully downloaded and saved {len(sample_list)} samples to: {output_path}")

if __name__ == "__main__":
    target_file = os.path.join("data", "english", "slim_orca.json")
    download_slim_orca_sample(target_file, num_samples=5000)