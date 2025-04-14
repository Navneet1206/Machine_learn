from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

class BusinessModelChatbot:
    def __init__(self):
        """Initialize the chatbot by loading the model, tokenizer, and training data embeddings."""
        print("Loading model and tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("models/business_model")
        self.model = DistilBertModel.from_pretrained("models/business_model")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("Model loaded successfully. Device set to use CPU.")
        print("Loading training data embeddings...")
        self.data_embeddings, self.data_sentences = self.load_training_data()
        print("Training data loaded.")

    def load_training_data(self):
        """Load and process training data from business_data.txt, computing embeddings for each sentence."""
        with open("data/processed/business_data.txt", "r") as f:
            lines = f.readlines()
        sentences = []
        for line in lines:
            line = line.strip()
            if line:
                # Split each line (page) into sentences
                page_sentences = [s.strip() for s in line.split('. ') if s.strip()]
                sentences.extend(page_sentences)
        embeddings = []
        for sent in sentences:
            embedding = self.get_embedding(sent)
            embeddings.append(embedding)
        return np.array(embeddings), sentences

    def get_embedding(self, text):
        """Generate an embedding for the given text using DistilBERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding from the last hidden state
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    def get_response(self, user_input):
        """Retrieve the most relevant response based on the user's input."""
        input_embedding = self.get_embedding(user_input)
        # Compute cosine similarity between input embedding and data embeddings
        dot_products = np.dot(self.data_embeddings, input_embedding)
        data_norms = np.linalg.norm(self.data_embeddings, axis=1)
        input_norm = np.linalg.norm(input_embedding)
        similarities = dot_products / (data_norms * input_norm + 1e-8)  # Avoid division by zero
        # Get top 3 most similar sentences
        top_indices = np.argsort(similarities)[-3:][::-1]
        responses = [self.data_sentences[i] for i in top_indices]
        return responses[0], responses  # Return best response and alternatives

    def start_chat(self):
        """Start the interactive chat session."""
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
                # Uncomment below to show alternative responses
                # print("Other suggestions:")
                # for i, alt in enumerate(alternatives[1:], 1):
                #     print(f"  {i+1}. {alt}")
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    chatbot = BusinessModelChatbot()
    chatbot.start_chat()