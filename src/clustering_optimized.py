import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedCategorizer:
    """Optimized categorizer with vectorized operations and compiled patterns."""
    
    def __init__(self):
        self.categories = self._load_categories()
        self.compiled_patterns = self._compile_patterns()
        self.voice_pattern = self._compile_voice_pattern()
        self.danish_pattern = self._compile_danish_pattern()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for performance monitoring."""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _load_categories(self) -> Dict:
        """Load and return category definitions."""
        return {
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
    
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for efficient matching."""
        compiled = {}
        
        for category, info in self.categories.items():
            # Compile main category patterns
            main_patterns = [re.escape(keyword) for keyword in info['keywords']]
            compiled[category] = {
                'main_pattern': re.compile(r'\b(?:' + '|'.join(main_patterns) + r')\b', re.IGNORECASE),
                'subcategories': {}
            }
            
            # Compile subcategory patterns
            for sub_name, sub_keywords in info['subcategories'].items():
                sub_patterns = [re.escape(keyword) for keyword in sub_keywords]
                compiled[category]['subcategories'][sub_name] = re.compile(
                    r'\b(?:' + '|'.join(sub_patterns) + r')\b', re.IGNORECASE
                )
        
        return compiled
    
    def _compile_voice_pattern(self) -> re.Pattern:
        """Compile voice detection pattern."""
        voice_keywords = ['transcript', 'audio', 'voice', 'speech', 'spoken', 'recording', 'sound']
        voice_patterns = [re.escape(keyword) for keyword in voice_keywords]
        return re.compile(r'\b(?:' + '|'.join(voice_patterns) + r')\b', re.IGNORECASE)
    
    def _compile_danish_pattern(self) -> re.Pattern:
        """Compile Danish language detection pattern."""
        danish_markers = ['hvordan', 'hvad', 'hvilken', 'hvor', 'hvem', 'hvorfor', 'skriv', 'hjælp', 'tak', 
                         'opslag', 'sæson', 'hold', 'med', 'og', 'eller', 'på', 'jeg', 'du', 'vi', 'denne']
        danish_patterns = [re.escape(marker) for marker in danish_markers]
        return re.compile(r'\b(?:' + '|'.join(danish_patterns) + r')\b', re.IGNORECASE)
    
    @lru_cache(maxsize=10000)
    def is_danish_text_cached(self, text: str) -> bool:
        """Cached Danish text detection."""
        matches = len(self.danish_pattern.findall(text))
        return matches >= 2
    
    @lru_cache(maxsize=10000)
    def is_voice_conversation_cached(self, text: str) -> bool:
        """Cached voice conversation detection."""
        return bool(self.voice_pattern.search(text))
    
    def get_category_scores_vectorized(self, texts: List[str]) -> List[Dict]:
        """Vectorized category scoring for multiple texts."""
        results = []
        
        for text in texts:
            text_lower = text.lower()
            scores = {}
            
            for category, patterns in self.compiled_patterns.items():
                # Count main category matches
                main_matches = len(patterns['main_pattern'].findall(text_lower))
                
                # Count subcategory matches
                sub_scores = {}
                total_sub_matches = 0
                for sub_name, sub_pattern in patterns['subcategories'].items():
                    sub_matches = len(sub_pattern.findall(text_lower))
                    sub_scores[sub_name] = sub_matches
                    total_sub_matches += sub_matches
                
                scores[category] = {
                    'main_score': main_matches,
                    'sub_scores': sub_scores,
                    'total_score': main_matches + total_sub_matches
                }
            
            results.append(scores)
        
        return results
    
    def assign_categories_batch(self, texts: List[str]) -> List[Dict]:
        """Batch category assignment for improved performance."""
        # Get all scores at once
        all_scores = self.get_category_scores_vectorized(texts)
        
        results = []
        for i, (text, scores) in enumerate(zip(texts, all_scores)):
            # Find best category
            best_category = max(scores.items(), key=lambda x: x[1]['total_score'])
            main_category = best_category[0]
            
            # Find best subcategory
            sub_scores = best_category[1]['sub_scores']
            best_subcategory = max(sub_scores.items(), key=lambda x: x[1]) if sub_scores else ('Uncategorized', 0)
            subcategory = best_subcategory[0]
            
            # Handle no matches case
            if best_category[1]['total_score'] == 0:
                main_category = 'Other'
                subcategory = 'Uncategorized'
            
            # Check voice and language
            is_voice = self.is_voice_conversation_cached(text)
            
            results.append({
                'main_category': main_category,
                'subcategory': subcategory,
                'is_voice': is_voice,
                'confidence_score': best_category[1]['total_score'],
                'all_scores': scores
            })
        
        return results

def calculate_metrics_optimized(df: pd.DataFrame) -> Dict:
    """Optimized metrics calculation using vectorized operations."""
    
    # Use vectorized operations where possible
    total_conversations = len(df)
    
    # Text length calculations
    text_lengths = df['text'].str.len()
    avg_length = text_lengths.mean()
    
    # Category distributions
    category_dist = df['main_category'].value_counts().to_dict()
    subcategory_dist = df['subcategory'].value_counts().to_dict()
    
    # Voice conversations
    voice_count = df['is_voice'].sum() if 'is_voice' in df.columns else 0
    
    # Complexity bins using cut for efficiency
    complexity_labels = ['simple', 'medium', 'complex']
    complexity_bins = [0, 100, 500, float('inf')]
    complexity_categories = pd.cut(text_lengths, bins=complexity_bins, labels=complexity_labels)
    complexity_counts = complexity_categories.value_counts().to_dict()
    
    # Interaction patterns using vectorized string operations
    question_marks = df['text'].str.count(r'\?').sum()
    follow_ups = df['text'].str.count(r'follow up|followup|additional', case=False).sum()
    
    return {
        'total_conversations': int(total_conversations),
        'avg_length': float(avg_length),
        'category_distribution': {k: int(v) for k, v in category_dist.items()},
        'subcategory_distribution': {k: int(v) for k, v in subcategory_dist.items()},
        'voice_conversations': int(voice_count),
        'complexity_scores': {k: int(v) for k, v in complexity_counts.items()},
        'interaction_patterns': {
            'questions': int(question_marks),
            'follow_ups': int(follow_ups)
        },
        'confidence_metrics': {
            'avg_confidence': float(df['confidence_score'].mean()) if 'confidence_score' in df.columns else 0,
            'high_confidence': int((df['confidence_score'] > 5).sum()) if 'confidence_score' in df.columns else 0
        }
    }

def run_clustering_optimized(cleaned_csv: str, output_csv: str, batch_size: int = 1000):
    """Optimized clustering with batch processing and performance monitoring."""
    
    start_time = time.time()
    categorizer = OptimizedCategorizer()
    
    # Load data
    categorizer.logger.info("Loading conversation data...")
    df = pd.read_csv(cleaned_csv)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' column")
    
    total_conversations = len(df)
    categorizer.logger.info(f"Processing {total_conversations} conversations in batches of {batch_size}")
    
    # Process in batches for memory efficiency
    all_results = []
    
    for i in range(0, total_conversations, batch_size):
        batch_end = min(i + batch_size, total_conversations)
        batch_texts = df['text'].iloc[i:batch_end].tolist()
        
        # Process batch
        batch_results = categorizer.assign_categories_batch(batch_texts)
        all_results.extend(batch_results)
        
        categorizer.logger.info(f"Processed batch {i//batch_size + 1}/{(total_conversations-1)//batch_size + 1}")
    
    # Add results to dataframe using vectorized operations
    result_df = pd.DataFrame(all_results)
    
    # Combine with original dataframe
    for col in ['main_category', 'subcategory', 'is_voice', 'confidence_score']:
        if col in result_df.columns:
            df[col] = result_df[col]
    
    # Add complexity metrics vectorized
    df['char_length'] = df['text'].str.len()
    df['complexity'] = pd.cut(df['char_length'], 
                            bins=[0, 100, 500, float('inf')],
                            labels=['simple', 'medium', 'complex'])
    
    # Calculate metrics
    categorizer.logger.info("Calculating metrics...")
    metrics = calculate_metrics_optimized(df)
    
    # Add performance metrics
    processing_time = time.time() - start_time
    metrics['performance'] = {
        'processing_time_seconds': processing_time,
        'conversations_per_second': total_conversations / processing_time,
        'batch_size_used': batch_size
    }
    
    # Save results
    categorizer.logger.info("Saving results...")
    
    # Save metrics
    metrics_file = os.path.join(os.path.dirname(output_csv), "conversation_metrics_optimized.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4, default=str)  # default=str handles numpy types
    
    # Save categorized conversations
    df.to_csv(output_csv, index=False)
    
    categorizer.logger.info(f"Categorization complete in {processing_time:.2f} seconds")
    categorizer.logger.info(f"Results saved to {output_csv}")
    categorizer.logger.info(f"Metrics saved to {metrics_file}")
    categorizer.logger.info(f"Processing speed: {metrics['performance']['conversations_per_second']:.1f} conversations/second")
    
    # Print summary statistics
    print_summary_optimized(df, metrics)

def print_summary_optimized(df: pd.DataFrame, metrics: Dict):
    """Print optimized summary with key statistics."""
    
    print("\n" + "="*50)
    print("CATEGORIZATION SUMMARY")
    print("="*50)
    
    print(f"Total conversations: {metrics['total_conversations']}")
    print(f"Processing time: {metrics['performance']['processing_time_seconds']:.2f} seconds")
    print(f"Speed: {metrics['performance']['conversations_per_second']:.1f} conversations/second")
    
    print("\nTop Categories:")
    for category, count in list(metrics['category_distribution'].items())[:10]:
        percentage = (count / metrics['total_conversations']) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\nVoice conversations: {metrics['voice_conversations']}")
    print(f"Average confidence score: {metrics['confidence_metrics']['avg_confidence']:.2f}")
    print(f"High confidence predictions: {metrics['confidence_metrics']['high_confidence']}")

if __name__ == "__main__":
    # Define file paths
    cleaned_csv = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    output_csv = os.path.join("..", "data", "processed", "categorized_conversations_optimized.csv")
    
    # Run optimized categorization
    run_clustering_optimized(cleaned_csv, output_csv)