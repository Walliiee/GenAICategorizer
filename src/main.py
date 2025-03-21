import os
import argparse
import pandas as pd
import json
from datetime import datetime

# Import modules
import categorization
import topic_modeling
import visualization

def ensure_directories():
    """Ensure all required directories exist"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of script directory)
    project_root = os.path.dirname(script_dir)
    
    dirs = [
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "data", "embeddings"),
        os.path.join(project_root, "data", "processed", "topic_modeling"),
        os.path.join(project_root, "data", "processed", "topic_modeling", "visualizations")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def run_complete_workflow(args):
    """Run the complete analysis workflow"""
    print("\n" + "="*80)
    print("Starting GenAI Conversation Analysis Workflow")
    print("="*80)
    
    # Ensure directories exist
    ensure_directories()
    
    # Get absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define file paths relative to project root
    cleaned_csv = args.input_file
    embedding_file = os.path.join(project_root, "data", "embeddings", "conversation_embeddings.npy")
    keyword_output_csv = os.path.join(project_root, "data", "processed", "keyword_categorized_conversations.csv")
    topic_output_dir = os.path.join(project_root, "data", "processed", "topic_modeling")
    visualization_dir = os.path.join(topic_output_dir, "visualizations")
    combined_output_csv = os.path.join(project_root, "data", "processed", "combined_results.csv")
    
    # 1. Keyword-based categorization
    print("\n" + "-"*40)
    print("Step 1: Running keyword-based categorization")
    print("-"*40)
    keyword_df, keyword_metrics = categorization.run_keyword_categorization(cleaned_csv, keyword_output_csv)
    
    # 2. Topic modeling
    print("\n" + "-"*40)
    print("Step 2: Running topic modeling")
    print("-"*40)
    topic_df, topic_model = topic_modeling.run_topic_modeling(
        input_csv=cleaned_csv,
        output_json=os.path.join(topic_output_dir, 'topic_data.json'),
        output_csv=os.path.join(topic_output_dir, 'topic_modeled_conversations.csv'),
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics
    )
    
    # Load topic data
    with open(os.path.join(topic_output_dir, 'topic_data.json'), 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    
    # 3. Create visualizations
    print("\n" + "-"*40)
    print("Step 3: Creating visualizations")
    print("-"*40)
    visualization.visualize_results(topic_df, topic_data, visualization_dir)
    
    # 4. Combine results
    print("\n" + "-"*40)
    print("Step 4: Combining results")
    print("-"*40)
    
    # Merge keyword-based and topic-based results
    keyword_df = keyword_df[['conversation_id', 'text', 'main_category', 'subcategory', 'is_voice', 'complexity']]
    
    # Ensure both dataframes have the conversation_id column for merging
    if 'conversation_id' in topic_df.columns and 'conversation_id' in keyword_df.columns:
        combined_df = pd.merge(
            keyword_df, 
            topic_df[['conversation_id', 'topic', 'topic_main_category', 'topic_subcategory', 'topic_confidence']], 
            on='conversation_id',
            how='outer'
        )
    else:
        # If no common ID column, assume they're in the same order
        print("Warning: No common ID column for merging. Assuming dataframes are in the same order.")
        combined_df = keyword_df.copy()
        combined_df['topic'] = topic_df['topic'].values
        combined_df['topic_main_category'] = topic_df['topic_main_category'].values
        combined_df['topic_subcategory'] = topic_df['topic_subcategory'].values
        combined_df['topic_confidence'] = topic_df['topic_confidence'].values
    
    # Add agreement column (do the categorizations agree?)
    combined_df['categories_agree'] = combined_df['main_category'] == combined_df['topic_main_category']
    
    # Save combined results
    combined_df.to_csv(combined_output_csv, index=False)
    print(f"Combined results saved to {combined_output_csv}")
    
    # 5. Calculate and save combined metrics
    combined_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'keyword_metrics': keyword_metrics,
        'topic_metrics': {
            'total_topics': len(topic_data['topic_info']) - 1,  # Exclude -1 (outlier) topic
            'topic_distribution': {str(k): int(v) for k, v in topic_df['topic'].value_counts().to_dict().items()},
            'category_distribution': {k: int(v) for k, v in topic_df['topic_main_category'].value_counts().to_dict().items()},
        },
        'agreement': {
            'total_agreement_rate': float(combined_df['categories_agree'].mean()),
            'agreement_by_keyword_category': {
                category: float(combined_df[combined_df['main_category'] == category]['categories_agree'].mean())
                for category in combined_df['main_category'].unique()
            }
        }
    }
    
    metrics_file = os.path.join(project_root, "data", "processed", "combined_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(combined_metrics, f, ensure_ascii=False, indent=4)
    
    print(f"Combined metrics saved to {metrics_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("Analysis Workflow Completed")
    print("="*80)
    print(f"Total conversations analyzed: {len(combined_df)}")
    print(f"Number of topics discovered: {combined_metrics['topic_metrics']['total_topics']}")
    print(f"Agreement rate between methods: {combined_metrics['agreement']['total_agreement_rate']:.2%}")
    print("\nTop 5 topics:")
    top_topics = topic_df['topic'].value_counts().head(5).to_dict()
    for topic, count in top_topics.items():
        topic_idx = str(int(topic))
        if topic_idx in topic_data['topic_mapping']:
            topic_name = topic_data['topic_mapping'][topic_idx]
            print(f"  - {topic_name}: {count} conversations")
    
    print("\nTop 5 categories from keyword method:")
    for category, count in keyword_df['main_category'].value_counts().head(5).items():
        print(f"  - {category}: {count} conversations")
    
    print("\nTop 5 categories from topic modeling:")
    for category, count in topic_df['topic_main_category'].value_counts().head(5).items():
        print(f"  - {category}: {count} conversations")
    
    print("\nOutput files:")
    print(f"  - Keyword categorization: {keyword_output_csv}")
    print(f"  - Topic modeling: {os.path.join(topic_output_dir, 'topic_modeled_conversations.csv')}")
    print(f"  - Visualizations: {visualization_dir}")
    print(f"  - Combined results: {combined_output_csv}")
    print(f"  - Metrics: {metrics_file}")

def parse_arguments():
    """Parse command line arguments"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of script directory)
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='GenAI Conversation Analysis')
    
    parser.add_argument('--input-file', type=str, 
                       default=os.path.join(project_root, "data", "processed", "cleaned_conversations.csv"),
                       help='Path to the cleaned conversation CSV file')
    
    parser.add_argument('--min-topic-size', type=int, default=10,
                       help='Minimum size of topics for BERTopic')
    
    parser.add_argument('--nr-topics', default="auto",
                       help='Number of topics to generate or "auto" for automatic detection')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_complete_workflow(args)
