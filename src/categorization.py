import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Get the script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load configuration from JSON files
categories_path = os.path.join(project_root, "config", "categories.json")
voice_keywords_path = os.path.join(project_root, "config", "voice_keywords.json")

# Load categories from JSON file
try:
    with open(categories_path, 'r', encoding='utf-8') as f:
        CATEGORIES = json.load(f)
    print(f"Loaded categories from {categories_path}")
except FileNotFoundError:
    print(f"Warning: Categories file not found at {categories_path}. Using default categories.")
    # Define main categories and their keywords/patterns as fallback
    CATEGORIES = {
        'Learning/Education': {
            'keywords': ['explain', 'teach', 'learn', 'understand', 'what is'],
            'subcategories': {
                'Concept Understanding': ['what is', 'define', 'concept', 'mean'],
                'How-to Learning': ['how to', 'steps', 'guide', 'tutorial']
            }
        },
        'Technical Support': {
            'keywords': ['help', 'fix', 'issue', 'problem', 'error'],
            'subcategories': {
                'Troubleshooting': ['troubleshoot', 'diagnose', 'fix', 'issue', 'error'],
                'Setup/Installation': ['setup', 'install', 'configure', 'connect']
            }
        }
    }

# Load voice keywords from JSON file
try:
    with open(voice_keywords_path, 'r', encoding='utf-8') as f:
        voice_config = json.load(f)
        VOICE_KEYWORDS = voice_config["VOICE_KEYWORDS"]
        DANISH_MARKERS = voice_config["DANISH_MARKERS"]
    print(f"Loaded voice keywords from {voice_keywords_path}")
except FileNotFoundError:
    print(f"Warning: Voice keywords file not found at {voice_keywords_path}. Using defaults.")
    # Define fallback voice keywords
    VOICE_KEYWORDS = ['transcript', 'audio', 'voice', 'speech', 'spoken', 'recording']
    DANISH_MARKERS = ['hvordan', 'hvad', 'hvilken', 'hvor', 'hvem', 'hvorfor']

def is_danish_text(text):
    """Check if text is likely Danish"""
    text_lower = text.lower()
    danish_word_count = sum(1 for marker in DANISH_MARKERS if marker in text_lower)
    return danish_word_count >= 2  # If 2 or more Danish markers are found

def is_voice_conversation(text):
    """Check if the conversation is voice-based"""
    text = text.lower()
    return any(keyword in text for keyword in VOICE_KEYWORDS)

def get_category_scores(text):
    """Get scores for each category based on keyword matching"""
    text = text.lower()
    scores = {}
    
    # Calculate keyword-based scores
    for category, info in CATEGORIES.items():
        # Main category score
        main_score = sum(1 for keyword in info['keywords'] if keyword in text)
        
        # Subcategory scores
        sub_scores = {}
        for sub_name, sub_keywords in info['subcategories'].items():
            sub_score = sum(1 for keyword in sub_keywords if keyword in text)
            sub_scores[sub_name] = sub_score
        
        scores[category] = {
            'main_score': main_score,
            'sub_scores': sub_scores,
            'total_score': main_score + sum(sub_scores.values())
        }
    
    return scores

def assign_categories(text):
    """Assign main category and subcategory based on keyword matching"""
    scores = get_category_scores(text)
    
    # Find the category with the highest total score
    main_category = max(scores.items(), key=lambda x: x[1]['total_score'])[0]
    
    # Find the subcategory with the highest score within the main category
    subcategory = max(scores[main_category]['sub_scores'].items(), 
                     key=lambda x: x[1])[0]
    
    # If no clear category is found (all scores are 0)
    if scores[main_category]['total_score'] == 0:
        main_category = 'Other'
        subcategory = 'Uncategorized'
    
    # Check if it's a voice conversation
    is_voice = is_voice_conversation(text)
    
    return {
        'main_category': main_category,
        'subcategory': subcategory,
        'is_voice': is_voice,
        'all_scores': scores
    }

def apply_keyword_categorization(df):
    """Apply keyword-based categorization to a dataframe"""
    # Assign categories and get detailed scores
    categorization_results = df['text'].apply(assign_categories)
    
    # Extract categories and scores
    df['main_category'] = categorization_results.apply(lambda x: x['main_category'])
    df['subcategory'] = categorization_results.apply(lambda x: x['subcategory'])
    df['is_voice'] = categorization_results.apply(lambda x: x['is_voice'])
    
    # Add complexity metrics
    df['char_length'] = df['text'].str.len()
    df['complexity'] = pd.cut(df['char_length'], 
                            bins=[0, 100, 500, float('inf')],
                            labels=['simple', 'medium', 'complex'])
    
    return df

def calculate_metrics(df):
    """Calculate useful metrics about the conversations"""
    metrics = {
        'total_conversations': int(len(df)),
        'avg_length': float(df['text'].str.len().mean()),
        'category_distribution': {k: int(v) for k, v in df['main_category'].value_counts().to_dict().items()},
        'subcategory_distribution': {k: int(v) for k, v in df['subcategory'].value_counts().to_dict().items()},
        'voice_conversations': int(df['is_voice'].sum()) if 'is_voice' in df.columns else 0,
        'complexity_scores': {
            'simple': int(len(df[df['text'].str.len() < 100])),
            'medium': int(len(df[(df['text'].str.len() >= 100) & (df['text'].str.len() < 500)])),
            'complex': int(len(df[df['text'].str.len() >= 500]))
        },
        'interaction_patterns': {
            'questions': int(df['text'].str.count(r'\?').sum()),
            'follow_ups': int(df['text'].str.count(r'follow up|followup|additional').sum())
        }
    }
    return metrics

def run_keyword_categorization(cleaned_csv, output_csv):
    """Process and categorize conversations using keywords"""
    # Load the cleaned CSV to retrieve conversation IDs and text
    df = pd.read_csv(cleaned_csv)
    
    # Apply categorization
    df = apply_keyword_categorization(df)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Save metrics to JSON
    metrics_file = os.path.join(os.path.dirname(output_csv), "keyword_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save the detailed dataframe
    df.to_csv(output_csv, index=False)
    print(f"Keyword categorization complete. Results saved to {output_csv}")
    print(f"Metrics saved to {metrics_file}")
    
    # Print summary
    print("\nCategory Distribution:")
    print(df['main_category'].value_counts())
    
    print("\nSubcategory Distribution:")
    for category in df['main_category'].unique():
        print(f"\n{category}:")
        subcats = df[df['main_category'] == category]['subcategory'].value_counts()
        print(subcats)
    
    print("\nVoice Conversations:")
    print(f"Total voice conversations: {df['is_voice'].sum()}")
    print("\nVoice conversations by category:")
    print(df[df['is_voice']]['main_category'].value_counts())
    
    return df, metrics

if __name__ == "__main__":
    # Define file paths
    cleaned_csv = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    output_csv = os.path.join("..", "data", "processed", "keyword_categorized_conversations.csv")
    
    # Run categorization
    run_keyword_categorization(cleaned_csv, output_csv) 