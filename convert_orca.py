import json
import os

def convert_orca_to_txt(json_input_path: str, txt_output_path: str, max_samples: int = 500, max_char_length: int = 300):
    """
    SlimOrca JSON format को Mini LLM readable standard format me convert karta hai.
    
    Format:
    User: <human_question>
    Assistant: <gpt_answer>
    """
    if not os.path.exists(json_input_path):
        print(f"File not found: {json_input_path}")
        return

    formatted_conversations = []
    
    with open(json_input_path, "r", encoding="utf-8") as f:
        # Load JSON data (list of conversations)
        try:
            data = json.load(f)
        except Exception:
            # Handling JSONL format (line by line)
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    count = 0
    for entry in data:
        # Extracted list of messages inside 'conversations' or root list
        conversations = entry.get("conversations", entry) if isinstance(entry, dict) else entry
        
        human_text = ""
        gpt_text = ""

        for msg in conversations:
            role = msg.get("from", "").lower()
            val = msg.get("value", "").strip()

            if role == "human":
                human_text = val
            elif role == "gpt":
                gpt_text = val

        # Filter out extremely long responses to fit our CPU context window
        if human_text and gpt_text:
            if len(human_text) <= max_char_length and len(gpt_text) <= max_char_length:
                formatted_block = f"User: {human_text}\nAssistant: {gpt_text}\n"
                formatted_conversations.append(formatted_block)
                count += 1

            if count >= max_samples:
                break

    # Save to output file inside data directory
    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(formatted_conversations))

    print(f"Successfully converted {len(formatted_conversations)} high-quality Orca Q&A pairs!")
    print(f"Saved formatted dataset to: {txt_output_path}")

if __name__ == "__main__":
    # convert_orca.py me input file path yeh rakhein:
    input_file = os.path.join("data", "english", "oo-labeled_correct.gpt4.sharegpt.jsonl")
    output_file = os.path.join("data", "english", "orca_instruction_dataset.txt")
    
    convert_orca_to_txt(input_file, output_file, max_samples=1000, max_char_length=350)