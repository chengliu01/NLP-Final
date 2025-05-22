from transformers import AutoTokenizer
import json
import os
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

LLAMA_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = None

try:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    print(f"Tokenizer '{LLAMA_MODEL_ID}' loaded successfully.")

except Exception as e:
    print(f"Error loading tokenizer '{LLAMA_MODEL_ID}': {e}")
    print("Please ensure 'transformers' is installed (`pip install transformers`)")
    print("and you are logged in via `huggingface-cli login` if required for the model.")


def count_simulation_tokens(data_list):

    global tokenizer
    if tokenizer is None:
        print("Error: Tokenizer is not loaded. Cannot count tokens.")
        return None

    if not isinstance(data_list, list):
        print("Error: Input must be a list of dictionaries.")
        return None

    total_tokens = 0
    entry_count = 0
    
    for item_index, item in enumerate(data_list):

        if not isinstance(item, dict):
            print(f"Warning: Skipping item at index {item_index} as it's not a dictionary.")
            continue
        current_tokens = 0
        simulation_entries = item.get("simulation")

        if not isinstance(simulation_entries, list):
            continue

        for entry_index, entry in enumerate(simulation_entries):
            if not isinstance(entry, dict):
                print(f"Warning: Skipping entry at index {entry_index} in item {item_index} as it's not a dictionary.")
                continue

            text_to_tokenize = ""
            original_response = entry.get("original_response", "").strip()

            if original_response:
                text_to_tokenize = original_response
            else:
                content = entry.get("content", "").strip()
                if content:
                    text_to_tokenize = content

            if text_to_tokenize:
                try:
                    tokens = tokenizer.encode(text_to_tokenize)
                    total_tokens += len(tokens)
                    entry_count += 1
                    current_tokens += len(tokens)
                except Exception as e:
                    print(f"Error tokenizing text at item {item_index}, entry {entry_index}: {e}")

    return round(total_tokens/entry_count if entry_count > 0 else 0, 2)


if __name__ == "__main__":
    json_file_path = '/Users/liucheng/Desktop/RL/role_rl/CoSER/exp/simulation/test_set_random40_gpt-4o-nlp_reasoning.json'

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
    elif tokenizer is None:
        print("Cannot proceed without a tokenizer.")
    else:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total_token_count = count_simulation_tokens(data)

            if total_token_count >= 0:
                print(f"\nTotal calculated tokens: {total_token_count}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
