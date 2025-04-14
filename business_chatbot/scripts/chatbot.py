from transformers import (
    pipeline,
    DistilBertTokenizer,
    DistilBertForMaskedLM
)
import torch


class BusinessModelChatbot:
    def __init__(self):
        # Load fine-tuned model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("models/business_model")
        self.model = DistilBertForMaskedLM.from_pretrained("models/business_model")

        # Set up fill-mask pipeline on CPU
        self.fill_mask = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # Use CPU
        )

    def format_input(self, user_input):
        """
        Formats the input string by adding a [MASK] token if it's missing.
        """
        user_input = user_input.strip()

        if len(user_input.split()) == 1:
            return f"What is {user_input} [MASK]?"

        elif "[MASK]" not in user_input:
            if user_input.endswith("?"):
                return user_input[:-1].strip() + " [MASK]?"
            else:
                return user_input.strip() + " [MASK]"

        return user_input

    def get_response(self, user_input):
        """
        Gets the top prediction for the masked input.
        Returns the best response and list of top 3 predictions.
        """
        masked_input = self.format_input(user_input)
        predictions = self.fill_mask(masked_input)

        responses = [
            p['sequence'].replace('[CLS] ', '').replace(' [SEP]', '')
            for p in predictions[:3]
        ]
        return responses[0], responses

    def start_chat(self):
        print("\n🤖 Business Model Chatbot: Ask me anything about business models!")
        print("💡 Tip: Try asking 'What is a revenue model?', or just type 'value proposition'")
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                print("👋 Chatbot: Goodbye!")
                break

            try:
                best_response, all_predictions = self.get_response(user_input)
                print(f"Chatbot: {best_response}")
                # Optional: show alternative suggestions
                # for i, alt in enumerate(all_predictions[1:], 1):
                #     print(f"  Alt {i}: {alt}")
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    chatbot = BusinessModelChatbot()
    chatbot.start_chat()
