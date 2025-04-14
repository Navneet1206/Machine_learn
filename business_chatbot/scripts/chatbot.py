from transformers import (
    pipeline,
    DistilBertTokenizer,
    DistilBertForMaskedLM
)
import torch


class BusinessModelChatbot:
    def __init__(self):
        print("Loading model and tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("models/business_model")
        self.model = DistilBertForMaskedLM.from_pretrained("models/business_model")

        # Initialize fill-mask pipeline
        self.fill_mask = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU only
        )
        print("Model loaded successfully. Device set to use CPU.")

    def format_input(self, user_input):
        """
        Convert user input to a prompt compatible with fill-mask.
        E.g., 'revenue' -> 'Revenue is [MASK].'
        """
        user_input = user_input.strip().rstrip("?").capitalize()
        return f"{user_input} is [MASK]."

    def get_response(self, user_input):
        """
        Generate the best predicted response from the masked input.
        """
        masked_input = self.format_input(user_input)
        predictions = self.fill_mask(masked_input)

        responses = [
            p['sequence'].replace('[CLS] ', '').replace(' [SEP]', '').replace('[MASK]', '').strip()
            for p in predictions[:3]
        ]
        top_response = responses[0]

        return top_response, responses

    def start_chat(self):
        print("\n🤖 Business Model Chatbot: Ask me anything about business models!")
        print("💡 Tip: Try asking 'What is a revenue model?', or just type 'value proposition'")
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("👋 Chatbot: Goodbye!")
                break

            try:
                best_response, alternatives = self.get_response(user_input)
                print(f"Chatbot: {best_response}")
                # Uncomment below if you want to show alternative suggestions
                # print("Other suggestions:")
                # for i, alt in enumerate(alternatives[1:], 1):
                #     print(f"  {i+1}. {alt}")
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    chatbot = BusinessModelChatbot()
    chatbot.start_chat()
