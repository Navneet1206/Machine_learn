from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch

def train_model():
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForMaskedLM.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset("text", data_files={'train': 'data/processed/business_data.txt'})
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    # Train and save the model
    trainer.train()
    model.save_pretrained("models/business_model")
    tokenizer.save_pretrained("models/business_model")

if __name__ == "__main__":
    train_model()