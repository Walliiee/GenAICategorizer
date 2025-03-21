import os
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
import matplotlib.pyplot as plt
import plotly.express as px
from umap import UMAP
from datetime import datetime
import logging
from categorization import CATEGORIES

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_or_create_embeddings(csv_file, embedding_file, model_name="paraphrase-multilingual-mpnet-base-v2", force_recreate=False):
    """Load existing embeddings or create new ones if needed"""
    # Check if embeddings already exist
    if os.path.exists(embedding_file) and not force_recreate:
        print(f"Loading embeddings from {embedding_file}")
        embeddings = np.load(embedding_file)
        df = pd.read_csv(csv_file)
        return df, embeddings
    
    # If not, create them
    print(f"Creating new embeddings using {model_name}...")
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Create embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save embeddings
    os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
    np.save(embedding_file, embeddings)
    
    print(f"Embeddings created and saved to {embedding_file}")
    return df, embeddings

def create_topic_model(df, embeddings, min_topic_size=10, nr_topics="auto", top_n_words=10):
    """
    Create a BERTopic model using the provided embeddings
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the conversation data
    embeddings : numpy array
        Pre-computed embeddings for the conversations
    min_topic_size : int
        Minimum size of topics
    nr_topics : int or "auto"
        Number of topics to generate or "auto" for automatic detection
    top_n_words : int
        Number of top words to use for representation
    
    Returns:
    --------
    topic_model : BERTopic
        Trained topic model
    topics : list
        List of topics assigned to each document
    probs : numpy array
        Topic probabilities for each document
    """
    # Create components for BERTopic
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.95)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired(top_n_words=top_n_words)
    
    # Create embedding model
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    # Initialize and fit BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        verbose=True
    )
    
    # Fit the model with pre-computed embeddings
    topics, probs = topic_model.fit_transform(df['text'], embeddings)
    
    return topic_model, topics, probs

def analyze_topics(topic_model, df, topics, output_dir):
    """
    Analyze and visualize the discovered topics
    
    Parameters:
    -----------
    topic_model : BERTopic
        Trained topic model
    df : pandas DataFrame
        DataFrame containing the conversation data  
    topics : list
        List of topics assigned to each document
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    topic_data : dict
        Dictionary with topic information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert topics to native Python integers if they're NumPy types
    topics = [int(t) if isinstance(t, np.integer) else t for t in topics]
    
    # Add topics to the dataframe
    df['topic'] = topics
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Get topic representations
    topic_representations = {}
    for topic in topic_info['Topic'].tolist():
        if topic != -1:  # Skip outlier topic
            words = [(str(word), float(score)) for word, score in topic_model.get_topic(topic)]
            topic_representations[str(int(topic))] = words
    
    # Create a topic mapping
    topic_mapping = {}
    for topic in topic_info['Topic'].tolist():
        if topic != -1:  # Skip outlier topic
            words = [word for word, _ in topic_model.get_topic(topic)]
            topic_mapping[str(int(topic))] = f"Topic {topic}: {', '.join(words[:5])}"
    
    # Get representative documents for each topic
    topic_docs = {}
    for topic in topic_info['Topic'].tolist():
        if topic != -1:  # Skip outlier topic
            docs = df[df['topic'] == topic]['text'].tolist()
            # Take a sample of up to 5 representative documents
            sample_size = min(5, len(docs))
            topic_docs[str(int(topic))] = docs[:sample_size]
    
    # Prepare data for visualization
    topic_distribution = df['topic'].value_counts().to_dict()
    topic_distribution = {str(k): int(v) for k, v in topic_distribution.items()}
    
    topic_data = {
        'topic_info': [
            {k: (int(v) if isinstance(v, np.integer) else 
                 float(v) if isinstance(v, np.floating) else v) 
             for k, v in record.items()}
            for record in topic_info.to_dict('records')
        ],
        'topic_representations': topic_representations,
        'topic_mapping': topic_mapping,
        'topic_docs': topic_docs,
        'topic_distribution': topic_distribution
    }
    
    # Save topic data
    with open(os.path.join(output_dir, 'topic_data.json'), 'w', encoding='utf-8') as f:
        json.dump(topic_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    
    # Create visualizations
    # 1. Topic word scores
    fig = topic_model.visualize_barchart(top_n_topics=10)
    fig.write_html(os.path.join(output_dir, "topic_barchart.html"))
    
    # 2. Topic similarity
    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(os.path.join(output_dir, "topic_hierarchy.html"))
    except Exception as e:
        print(f"Could not generate topic hierarchy visualization: {e}")
    
    # 3. Topic distribution
    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, "topic_visualization.html"))
    except Exception as e:
        print(f"Could not generate topic visualization: {e}")
    
    # 4. Topic heatmap
    try:
        fig = topic_model.visualize_heatmap()
        fig.write_html(os.path.join(output_dir, "topic_heatmap.html"))
    except Exception as e:
        print(f"Could not generate topic heatmap: {e}")
    
    print(f"Topic analysis complete. Results saved to {output_dir}")
    
    return topic_data

def map_topics_to_categories(topic_data, categories_dict, output_dir):
    """
    Map discovered topics to predefined categories
    
    Parameters:
    -----------
    topic_data : dict
        Dictionary with topic information from analyze_topics function
    categories_dict : dict
        Dictionary of predefined categories from categorization.py
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    category_mapping : dict
        Mapping of topics to categories
    """
    category_mapping = {}
    
    # For each topic, calculate similarity to each category
    for topic_id, words_with_scores in topic_data['topic_representations'].items():
        topic_words = [word for word, _ in words_with_scores]
        
        # Calculate score for each category
        category_scores = {}
        for category, info in categories_dict.items():
            # Calculate overlap with category keywords
            keywords = info['keywords']
            overlap = sum(1 for word in topic_words if any(keyword in word.lower() for keyword in keywords))
            
            # Calculate overlap with subcategory keywords
            subcategory_overlap = 0
            for _, subcategory_keywords in info['subcategories'].items():
                subcategory_overlap += sum(1 for word in topic_words 
                                          if any(keyword in word.lower() for keyword in subcategory_keywords))
            
            # Combined score
            category_scores[category] = overlap + subcategory_overlap
        
        # Assign to category with highest score
        if max(category_scores.values()) > 0:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            # Find best subcategory
            subcategory_scores = {}
            for subcategory, subcategory_keywords in categories_dict[best_category]['subcategories'].items():
                overlap = sum(1 for word in topic_words 
                             if any(keyword in word.lower() for keyword in subcategory_keywords))
                subcategory_scores[subcategory] = overlap
            
            best_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])[0] if max(subcategory_scores.values()) > 0 else "Other"
            
            category_mapping[topic_id] = {
                'main_category': best_category,
                'subcategory': best_subcategory,
                'confidence': category_scores[best_category] / sum(category_scores.values()) if sum(category_scores.values()) > 0 else 0
            }
        else:
            category_mapping[topic_id] = {
                'main_category': 'Other',
                'subcategory': 'Uncategorized',
                'confidence': 0
            }
    
    # Save the mapping
    with open(os.path.join(output_dir, 'topic_to_category_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, ensure_ascii=False, indent=4)
    
    return category_mapping

def apply_topic_categorization(df, topics, topic_to_category_mapping):
    """
    Apply topic-based categorization to a dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the conversation data
    topics : list
        List of topics assigned to each document
    topic_to_category_mapping : dict
        Mapping of topics to categories
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added topic-based categorization
    """
    # Add topics to the dataframe
    df['topic'] = topics
    
    # Convert topic to string for mapping lookup
    def get_category(topic_id):
        # Convert to string since our mapping uses string keys
        str_topic_id = str(int(topic_id)) if topic_id != -1 else "-1"
        if str_topic_id in topic_to_category_mapping:
            return topic_to_category_mapping[str_topic_id]['main_category']
        return 'Other'
    
    def get_subcategory(topic_id):
        # Convert to string since our mapping uses string keys
        str_topic_id = str(int(topic_id)) if topic_id != -1 else "-1"
        if str_topic_id in topic_to_category_mapping:
            return topic_to_category_mapping[str_topic_id]['subcategory']
        return 'Uncategorized'
    
    def get_confidence(topic_id):
        # Convert to string since our mapping uses string keys
        str_topic_id = str(int(topic_id)) if topic_id != -1 else "-1"
        if str_topic_id in topic_to_category_mapping:
            return topic_to_category_mapping[str_topic_id]['confidence']
        return 0
    
    # Add categories based on topic mapping
    df['topic_main_category'] = df['topic'].apply(
        lambda x: get_category(x) if x != -1 else 'Outlier'
    )
    
    df['topic_subcategory'] = df['topic'].apply(
        lambda x: get_subcategory(x) if x != -1 else 'Outlier'
    )
    
    df['topic_confidence'] = df['topic'].apply(
        lambda x: get_confidence(x) if x != -1 else 0
    )
    
    return df

def run_topic_modeling(input_csv, output_json, output_csv, min_topic_size=10, nr_topics="auto"):
    """
    Run the complete topic modeling workflow
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file with conversations
    output_json : str
        Path to save topic data JSON
    output_csv : str
        Path to save topic-modeled conversations CSV
    min_topic_size : int
        Minimum size of topics
    nr_topics : int or "auto"
        Number of topics to generate or "auto" for automatic detection
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with topic assignments
    topic_model : BERTopic
        Trained topic model
    """
    # Get directory for embeddings
    embedding_file = os.path.join(os.path.dirname(output_json), "embeddings.npy")
    
    # Load or create embeddings
    df, embeddings = load_or_create_embeddings(input_csv, embedding_file)
    
    # Create and train topic model
    topic_model, topics, probs = create_topic_model(
        df, embeddings, 
        min_topic_size=min_topic_size,
        nr_topics=nr_topics
    )
    
    # Analyze topics and save results
    output_dir = os.path.dirname(output_json)
    topic_data = analyze_topics(topic_model, df, topics, output_dir)
    
    # Map topics to categories using the predefined categories from categorization.py
    topic_to_category_mapping = map_topics_to_categories(topic_data, CATEGORIES, output_dir)
    
    # Apply the category mapping to the dataframe
    df = apply_topic_categorization(df, topics, topic_to_category_mapping)
    
    # Add main_category and subcategory columns as aliases for topic_main_category and topic_subcategory
    # This ensures compatibility with the visualization code
    df['main_category'] = df['topic_main_category']
    df['subcategory'] = df['topic_subcategory']
    
    # Print summary
    print("\nTopic Modeling Results:")
    print(f"Number of topics: {len(topic_data['topic_representations'])}")
    print("\nTop topics:")
    for topic_id, words in list(topic_data['topic_representations'].items())[:5]:
        print(f"Topic {topic_id}: {', '.join([word for word, _ in words[:5]])}")
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_json}")
    print(f"Categorized conversations saved to: {output_csv}")
    
    return df, topic_model

if __name__ == "__main__":
    import categorization
    
    # Define file paths
    cleaned_csv = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    embedding_file = os.path.join("..", "data", "embeddings", "conversation_embeddings.npy")
    output_dir = os.path.join("..", "data", "processed", "topic_modeling")
    
    # Run topic modeling
    run_topic_modeling(cleaned_csv, os.path.join(output_dir, "topic_modeling_results.json"), os.path.join(output_dir, "topic_modeled_conversations.csv"), categorization.CATEGORIES) 