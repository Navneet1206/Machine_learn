import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embedding(self, text):
        return self.model.encode(text)

class BusinessModelChatbot:
    def __init__(self):
        self.console = Console()
        self.embedder = EmbeddingModel()
        self.chat_history = []
        self.console.print("🤖 [bold green]Initializing Chatbot...[/bold green]")
        self.data_embeddings, self.data_sentences = self.load_training_data()
        self.console.print("✅ [bold cyan]Chatbot Ready![/bold cyan]")

    def load_training_data(self):
        file_path = "D:/Downloads-folder/GPT Help/Machine_learn/business_chatbot/data/processed/business_data.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing training data at {file_path}")

        with open(file_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        sentences = []
        for line in lines:
            page_sentences = [s.strip() for s in line.split('. ') if s.strip()]
            sentences.extend(page_sentences)

        embeddings = [self.embedder.get_embedding(sent) for sent in sentences]
        return np.array(embeddings), sentences

    def get_response(self, user_input):
        self.chat_history.append(user_input)
        context_input = " ".join(self.chat_history[-3:])
        input_embedding = self.embedder.get_embedding(context_input)

        similarities = np.dot(self.data_embeddings, input_embedding) / (
            np.linalg.norm(self.data_embeddings, axis=1) * np.linalg.norm(input_embedding) + 1e-8
        )
        top_indices = np.argsort(similarities)[-3:][::-1]
        best_response = self.data_sentences[top_indices[0]]
        return best_response

    def start_chat(self):
        self.console.print("\n🤖 [bold green]Business Model Chatbot[/bold green] at your service!")
        self.console.print("💡 Ask me about business models. Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                self.console.print("👋 Goodbye!")
                break
            try:
                response = self.get_response(user_input)
                self.console.print(f"[bold blue]Chatbot:[/bold blue] {response}")
            except Exception as e:
                self.console.print(f"[red]⚠️ Error: {str(e)}[/red]")

if __name__ == "__main__":
    bot = BusinessModelChatbot()
    bot.start_chat()
