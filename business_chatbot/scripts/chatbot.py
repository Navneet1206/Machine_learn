from transformers import (
    pipeline,
    DistilBertTokenizer,
    DistilBertForMaskedLM
)
import torch

class BusinessModelChatbot:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("models/business_model")
        self.model = DistilBertForMaskedLM.from_pretrained("models/business_model")
        self.fill_mask = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # Use CPU
        )
    
    def get_response(self, user_input):
        if "[MASK]" not in user_input:
            # Add a mask if none provided
            user_input = user_input.replace("?", " [MASK]?")
        
        predictions = self.fill_mask(user_input)
        return predictions[0]['sequence'].replace('[CLS] ', '').replace(' [SEP]', '')
    
    def start_chat(self):
        print("Business Model Chatbot: Ask me about business models! (Type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = BusinessModelChatbot()
    chatbot.start_chat()