import os
import warnings

# Import from new modules
from categorization import run_keyword_categorization, CATEGORIES

def run_clustering(cleaned_csv, output_csv):
    """Process and categorize conversations"""
    warnings.warn(
        "This module is deprecated. Please use the new modules: "
        "categorization.py, topic_modeling.py, and main.py instead.",
        DeprecationWarning, stacklevel=2
    )
    
    # Run the keyword-based categorization from the new module
    run_keyword_categorization(cleaned_csv, output_csv)

if __name__ == "__main__":
    # Define file paths
    cleaned_csv = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    output_csv = os.path.join("..", "data", "processed", "categorized_conversations.csv")
    
    # Show deprecation warning
    print("WARNING: This script is deprecated. Please use 'python main.py' instead.")
    
    # Run categorization
    run_clustering(cleaned_csv, output_csv)
    