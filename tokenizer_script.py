from transformers import GPT2TokenizerFast
import torch
from huggingface_hub import login

# 🔹 Authenticate with Hugging Face
login(token="")  # Replace with your actual token

# 🔹 Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token

# Tokenize text file
with open('data/cleaned_text1.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 🔹 Split into chunks of 512 tokens with padding
inputs = tokenizer(
    text,
    return_tensors='pt',
    truncation=True,
    max_length=512,
    stride=256,  # Overlap between chunks
    return_overflowing_tokens=True,
    padding=True  # Ensure padding is applied
)

# 🔹 Save tokenized data (input_ids and attention_mask separately)
torch.save(inputs['input_ids'], 'data/input_ids.pt')
torch.save(inputs['attention_mask'], 'data/attention_mask.pt')

print("✅ Tokenized data saved successfully.")

