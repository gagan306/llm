from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('my_custom_model')
tokenizer = GPT2Tokenizer.from_pretrained('my_custom_model')

# Define the question you want to ask
question = "what is  operating system?"

# Create the input prompt by combining the question with any context (if needed)
input_prompt = f"Q: {question} A:"

# Tokenize the input prompt
inputs = tokenizer.encode(input_prompt, return_tensors='pt')

# Generate the response
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True, top_p=0.9, top_k=50)

# Decode the generated response
generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated answer
print(f"Question: {question}")
print(f"Answer: {generated_answer[len(input_prompt):].strip()}")
