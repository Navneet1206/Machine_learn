import os
import re
from datasets import load_dataset

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.lower().strip()

def preprocess_wikipedia():
    # Specify a different storage path (e.g., D drive)
    cache_dir = "D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/data"  # Change this to your desired path
    
    dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir, trust_remote_code=True)
    
    business_pages = dataset.filter(
        lambda x: any(term in x['title'].lower() for term in 
                     ['business model', 'revenue', 'startup', 'corporate strategy'])
    )
    
    processed_text = []
    for page in business_pages:
        cleaned = clean_text(page['text'])
        processed_text.append(cleaned)
    
    # Ensure the output directory exists
    output_dir = "D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/data/processed"  # Change this to your output path
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'business_data.txt'), 'w') as f:
        f.write('\n'.join(processed_text))

if __name__ == "__main__":
    preprocess_wikipedia()