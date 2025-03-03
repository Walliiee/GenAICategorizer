import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(csv_file, output_file, model_name="paraphrase-multilingual-mpnet-base-v2"):
    # Load the cleaned conversation CSV file
    df = pd.read_csv(csv_file)
    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")
    
    # Convert the text column to string to ensure all entries are strings
    texts = df["text"].astype(str).tolist()
    
    # Load the pre-trained multilingual Sentence Transformer model
    print(f"Loading multilingual model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for each conversation text
    print(f"Generating language-agnostic embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Save the embeddings to a .npy file
    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    # Paths are relative to the src folder
    csv_file = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    output_file = os.path.join("..", "data", "processed", "embeddings.npy")
    
    generate_embeddings(csv_file, output_file)
