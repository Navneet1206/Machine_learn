import os
import logging
from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Features, Value

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    # Define paths
    data_dir = "D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/data/processed"
    model_dir = "D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/models/business_model"
    data_file = os.path.join(data_dir, 'business_data.txt')

    # Check if data file exists and is not empty
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    if os.path.getsize(data_file) == 0:
        logger.error(f"Data file is empty: {data_file}")
        return
    logger.info(f"Data file found: {data_file}")

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForMaskedLM.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer and model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        return

    # Load dataset with explicit schema
    try:
        # Define schema to avoid inference errors
        features = Features({"text": Value("string")})
        dataset = load_dataset(
            "text",
            data_files={'train': data_file},
            features=features,
            split="train"
        )
        logger.info(f"Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return

    # Check if dataset is empty
    if len(dataset) == 0:
        logger.error("Dataset is empty. Please check business_data.txt.")
        return

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    try:
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        logger.info("Tokenization complete")
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        return

    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/models",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Increased for better training
        per_device_train_batch_size=8,  # Adjusted for stability
        gradient_accumulation_steps=2,  # Simulate larger batch size
        save_steps=1000,
        save_total_limit=2,
        logging_dir="D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/logs",
        logging_steps=500,
        learning_rate=2e-5,  # Added for better convergence
        remove_unused_columns=False,
    )

    # Initialize Trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
        )
        logger.info("Trainer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {str(e)}")
        return

    # Train and save
    try:
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Training complete. Model saved to {model_dir}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    train_model()