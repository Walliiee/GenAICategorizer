import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import modules
import categorization
import topic_modeling
import visualization

def create_sample_dataset():
    """Create a small sample dataset for demonstration"""
    print("Creating sample dataset...")
    
    # Create sample dataframe
    sample_data = [
        {
            "conversation_id": "1",
            "text": "How do I write code to debug this Python error in my function?"
        },
        {
            "conversation_id": "2",
            "text": "Can you help me debug this JavaScript code? I'm getting a null reference error."
        },
        {
            "conversation_id": "3",
            "text": "I need to fix a bug in my Python program. It's crashing when I input large numbers."
        },
        {
            "conversation_id": "4",
            "text": "What's wrong with my code? I'm getting an error when I try to import this library."
        },
        {
            "conversation_id": "5",
            "text": "Can you help me draft an email to my client about the project timeline?"
        },
        {
            "conversation_id": "6",
            "text": "Write a professional email for a job application to a software company."
        },
        {
            "conversation_id": "7",
            "text": "Help me draft a cover letter for this senior developer position."
        },
        {
            "conversation_id": "8",
            "text": "Can you write a thank you email after an interview with a tech company?"
        },
        {
            "conversation_id": "9",
            "text": "Explain the concept of machine learning and how it works."
        },
        {
            "conversation_id": "10",
            "text": "What is deep learning and how does it differ from traditional machine learning?"
        },
        {
            "conversation_id": "11",
            "text": "Can you teach me about neural networks and their applications?"
        },
        {
            "conversation_id": "12",
            "text": "What is the difference between supervised and unsupervised learning in AI?"
        },
        {
            "conversation_id": "13",
            "text": "Hvordan kan jeg lære at kode i Python? Jeg er nybegynder."
        },
        {
            "conversation_id": "14",
            "text": "Forklar mig hvordan man bruger funktioner i JavaScript."
        },
        {
            "conversation_id": "15",
            "text": "Here's the transcript of our voice conversation about project management."
        }
    ]
    
    # Create dataframe
    df = pd.DataFrame(sample_data)
    
    # Create directories
    os.makedirs(os.path.join("..", "sample"), exist_ok=True)
    
    # Save to CSV
    sample_csv = os.path.join("..", "sample", "sample_conversations.csv")
    df.to_csv(sample_csv, index=False)
    
    print(f"Sample dataset created with {len(df)} conversations and saved to {sample_csv}")
    return sample_csv

def run_sample_analysis():
    """Run analysis on the sample dataset"""
    # Create sample dataset
    sample_csv = create_sample_dataset()
    
    # Set up file paths
    sample_dir = os.path.dirname(sample_csv)
    embedding_file = os.path.join(sample_dir, "sample_embeddings.npy")
    keyword_output_csv = os.path.join(sample_dir, "sample_keyword_categorized.csv")
    topic_output_dir = os.path.join(sample_dir, "topic_modeling")
    visualization_dir = os.path.join(topic_output_dir, "visualizations")
    
    # Ensure directories exist
    os.makedirs(topic_output_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 1. Run keyword-based categorization
    print("\n" + "-"*40)
    print("Step 1: Running keyword-based categorization")
    print("-"*40)
    keyword_df, keyword_metrics = categorization.run_keyword_categorization(sample_csv, keyword_output_csv)
    
    # 2. Run topic modeling
    print("\n" + "-"*40)
    print("Step 2: Running topic modeling")
    print("-"*40)
    
    # Create output file paths
    topic_output_json = os.path.join(topic_output_dir, "topic_modeling_results.json")
    topic_output_csv = os.path.join(topic_output_dir, "topic_modeled_conversations.csv")
    
    # Run topic modeling with the new function signature
    topic_modeling.run_topic_modeling(
        input_csv=sample_csv,
        output_json=topic_output_json,
        output_csv=topic_output_csv,
        min_topic_size=2,  # Small dataset, so use small minimum topic size
        nr_topics=3        # For demonstration, use 3 topics
    )
    
    # Load topic data
    with open(topic_output_json, 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    
    # Load the categorized data
    topic_df = pd.read_csv(topic_output_csv)
    
    # Load keyword-based categorization
    keyword_df = pd.read_csv(keyword_output_csv)
    
    # Merge the two categorizations
    combined_df = pd.merge(
        topic_df,
        keyword_df[['conversation_id', 'main_category', 'subcategory']],
        on='conversation_id',
        how='left'
    )
    
    # 3. Create visualizations
    print("\n" + "-"*40)
    print("Step 3: Creating visualizations")
    print("-"*40)
    visualization.visualize_results(combined_df, topic_data, visualization_dir)
    
    # 4. Print summary
    print("\n" + "="*80)
    print("Sample Analysis Complete")
    print("="*80)
    
    print("\nKeyword-based categorization:")
    print(keyword_df[['conversation_id', 'main_category', 'subcategory']].to_string(index=False))
    
    print("\nTopic-based categorization:")
    print(topic_df[['conversation_id', 'topic', 'topic_main_category', 'topic_subcategory']].to_string(index=False))
    
    print("\nTopic keywords:")
    for topic, words in topic_data['topic_keywords'].items():
        print(f"Topic {topic}: {', '.join(words)}")
    
    print("\nOutput files:")
    print(f"  - Keyword categorization: {keyword_output_csv}")
    print(f"  - Topic modeling: {topic_output_csv}")
    print(f"  - Visualizations: {visualization_dir}")
    
    print("\nYou can now explore the visualization files in your browser.")

if __name__ == "__main__":
    run_sample_analysis() 