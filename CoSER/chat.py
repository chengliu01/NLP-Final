from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify local model path
model_path = "Neph0s/CoSER-Llama-3.1-70B"

# Load local model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for large models to reduce memory usage
    device_map="auto"  # Automatically allocate device (CPU/GPU)
)

# Load local tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def chat_with_model(max_turns=5, max_new_tokens=128):
    system_prompt = input("System Prompt for Roleplay: ")

    messages = [{"role": "system", "content": system_prompt}]
    print("Starting Conversation. Type 'exit' to quit")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Character: Goodbye!")
            break
        
        if not user_input:  # Skip empty inputs
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Process input text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare model inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
            do_sample=True,  # Enable sampling for more natural responses
            temperature=0.7  # Control randomness in generation
        )
        
        # Filter out input tokens, keep only the generated part
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Character: {response}")
        
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat_with_model()