from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2TokenizerFast
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

# Load tokenized data (input_ids and attention_mask)
input_ids = torch.load('data/input_ids.pt')
attention_mask = torch.load('data/attention_mask.pt')

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Initialize model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token

# Convert the tokenized data into a dataset
train_dataset = TextDataset(input_ids, attention_mask)

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,  # Reduce if you get OOM errors
    num_train_epochs=1,
    save_steps=500,
    logging_steps=100,
    fp16=True,  # Enable if you have a GPU
)

# Data collator for language modeling (handles padding, etc.)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 does not use MLM (masked language modeling)
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator  # Use the data collator
)

# Start training
trainer.train()

# Save model and tokenizer
model.save_pretrained('my_custom_model')
tokenizer.save_pretrained('my_custom_model')

print("✅ Training complete and model saved.")
