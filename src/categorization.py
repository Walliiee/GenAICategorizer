import os
import pandas as pd
import numpy as np
import json

# Define main categories and their keywords/patterns
CATEGORIES = {
    'Learning/Education': {
        'keywords': ['explain', 'teach', 'learn', 'understand', 'what is', 'how does', 'define', 'example', 'tutorial',
                    'forklar', 'forklaring', 'forstå', 'lær', 'hvordan'],
        'subcategories': {
            'Concept Understanding': ['what is', 'define', 'concept', 'mean', 'forklar', 'hvad er'],
            'How-to Learning': ['how to', 'steps', 'guide', 'tutorial', 'hvordan'],
            'Problem Solving': ['solve', 'solution', 'answer', 'help me understand'],
            'Academic Topics': ['method', 'theory', 'model', 'metode', 'teori', 'analyse']
        }
    },
    'Code Development': {
        'keywords': ['code', 'program', 'function', 'debug', 'error', 'python', 'javascript', 'api', 'database', 'git'],
        'subcategories': {
            'Bug Fixing': ['error', 'bug', 'fix', 'issue', 'debug'],
            'Feature Development': ['create', 'implement', 'develop', 'add feature'],
            'Code Review': ['review', 'improve', 'optimize', 'refactor']
        }
    },
    'Writing Assistance': {
        'keywords': ['write', 'draft', 'review', 'proofread', 'text', 'article', 'edit', 'grammar'],
        'subcategories': {
            'Content Creation': ['write', 'create', 'draft'],
            'Editing': ['edit', 'proofread', 'review', 'improve'],
            'Format/Style': ['format', 'style', 'structure']
        }
    },
    'Analysis/Research': {
        'keywords': ['analyze', 'research', 'study', 'investigate', 'compare', 'evaluate', 'data', 'findings'],
        'subcategories': {
            'Data Analysis': ['data', 'statistics', 'numbers', 'trends'],
            'Research Review': ['research', 'paper', 'study', 'literature'],
            'Comparative Analysis': ['compare', 'difference', 'versus', 'pros and cons']
        }
    },
    'Creative & Ideation': {
        'keywords': ['create', 'design', 'generate', 'brainstorm', 'imagine', 'creative', 'innovative', 'idea', 
                    'suggest', 'think of', 'come up with', 'possibilities', 'concept'],
        'subcategories': {
            'Visual Design': ['design', 'visual', 'layout', 'look'],
            'Idea Generation': ['brainstorm', 'ideas', 'possibilities', 'suggest'],
            'Creative Problem Solving': ['solution', 'solve', 'address', 'improve'],
            'Innovation': ['innovative', 'new', 'unique', 'original'],
            'Concept Development': ['develop', 'refine', 'enhance', 'iterate']
        }
    },
    'Professional/Business': {
        'keywords': ['business', 'professional', 'company', 'client', 'strategy', 'market', 'industry'],
        'subcategories': {
            'Strategy': ['strategy', 'plan', 'approach'],
            'Client/Customer': ['client', 'customer', 'service'],
            'Business Analysis': ['analysis', 'market', 'industry']
        }
    },
    'Technical Support': {
        'keywords': ['help', 'fix', 'issue', 'problem', 'support', 'error', 'troubleshoot', 'sync', 'synch', 'calendar',
                    'setup', 'connect', 'integration', 'configure', 'settings', 'apple', 'microsoft', 'teams', 'outlook'],
        'subcategories': {
            'Troubleshooting': ['troubleshoot', 'diagnose', 'fix', 'issue', 'error'],
            'Setup/Installation': ['setup', 'install', 'configure', 'connect', 'sync'],
            'Usage Help': ['how to use', 'help with', 'guide', 'how do i'],
            'Integration Issues': ['sync', 'connect', 'integration', 'between', 'with']
        }
    },
    'Personal Projects': {
        'keywords': ['project', 'personal', 'help me with', 'my own', 'portfolio', 'hobby'],
        'subcategories': {
            'Project Planning': ['plan', 'structure', 'organize'],
            'Implementation': ['build', 'create', 'develop'],
            'Review/Feedback': ['review', 'feedback', 'improve']
        }
    },
    'SoMe/Marketing': {
        'keywords': ['social media', 'marketing', 'post', 'content', 'campaign', 'LinkedIn', 'Twitter', 'engagement',
                    'opslag', 'sæson', 'hold', 'announce', 'announcement', 'season', 'team', 'group', 'start'],
        'subcategories': {
            'Content Creation': ['post', 'content', 'create', 'opslag', 'write post'],
            'Event Announcements': ['season', 'sæson', 'event', 'start', 'announce', 'new'],
            'Team/Group Updates': ['team', 'hold', 'group', 'members', 'community'],
            'Campaign Planning': ['campaign', 'strategy', 'plan', 'theme', 'tema']
        }
    },
    'DALL-E/Image': {
        'keywords': ['image', 'picture', 'photo', 'generate image', 'create image', 'dall-e', 'dalle', 
                    'draw', 'illustration', 'visual', 'artwork'],
        'subcategories': {
            'Image Generation': ['generate', 'create', 'make'],
            'Image Editing': ['edit', 'modify', 'adjust'],
            'Style Transfer': ['style', 'artistic', 'filter'],
            'Visual Description': ['describe', 'detail', 'explain image']
        }
    },
    'Cooking/Food': {
        'keywords': ['food', 'cook', 'recipe', 'ingredient', 'meal', 'dish', 'kitchen', 'bake', 'pumpkin',
                    'vegetable', 'fruit', 'squash', 'cuisine', 'eat', 'dinner', 'lunch', 'breakfast'],
        'subcategories': {
            'Recipe Help': ['recipe', 'how to make', 'cook', 'bake', 'prepare'],
            'Ingredient Questions': ['ingredient', 'what is', 'substitute', 'alternative'],
            'Food Identification': ['what is this', 'identify', 'which', 'type of', 'variety'],
            'Meal Planning': ['meal', 'plan', 'menu', 'dinner', 'lunch', 'breakfast']
        }
    },
    'Information/Curiosity': {
        'keywords': ['what is', 'where is', 'why does', 'how does', 'tell me about', 'find information', 
                    'information about', 'can you find', 'explain why', 'hvorfor', 'hvor', 'hvad', 
                    'find ud af', 'finde information', 'information om'],
        'subcategories': {
            'General Knowledge': ['what is', 'tell me about', 'explain', 'fortæl'],
            'Location/Place Questions': ['where is', 'location', 'place', 'hvor'],
            'Cause/Effect': ['why does', 'how does', 'what causes', 'hvorfor'],
            'Research Requests': ['find information', 'research', 'look up', 'find ud af', 'søg'],
            'Industry/Topic Research': ['industry', 'sector', 'field', 'branche', 'område', 'digitalization', 'digitalisering']
        }
    }
}

# Special cross-cutting category for voice
VOICE_KEYWORDS = ['transcript', 'audio', 'voice', 'speech', 'spoken', 'recording', 'sound']

# Danish language markers to help with categorization
DANISH_MARKERS = ['hvordan', 'hvad', 'hvilken', 'hvor', 'hvem', 'hvorfor', 'skriv', 'hjælp', 'tak', 
                 'opslag', 'sæson', 'hold', 'med', 'og', 'eller', 'på', 'jeg', 'du', 'vi', 'denne']

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