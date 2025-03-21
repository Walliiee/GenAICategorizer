import os
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io

def create_topic_distribution_visualization(df, output_dir):
    """
    Create a visualization of topic distribution
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with topic modeling results
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a topic distribution chart
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    topic_counts = topic_counts[topic_counts['Topic'] != -1]  # Remove outlier topic
    topic_counts = topic_counts.sort_values('Count', ascending=False)
    
    fig = px.bar(topic_counts, x='Topic', y='Count', 
                title='Distribution of Topics',
                labels={'Topic': 'Topic Number', 'Count': 'Number of Conversations'},
                color='Count',
                color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_title="Topic Number",
        yaxis_title="Number of Conversations",
        font=dict(size=12)
    )
    
    fig.write_html(os.path.join(output_dir, "topic_distribution.html"))
    print(f"Topic distribution visualization saved to {os.path.join(output_dir, 'topic_distribution.html')}")

def create_category_comparison_visualization(df, output_dir):
    """
    Create a visualization comparing keyword-based vs. topic-based categorization
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with both keyword-based and topic-based categorization
    output_dir : str
        Directory to save visualizations
    """
    if not all(col in df.columns for col in ['main_category', 'topic_main_category']):
        print("DataFrame does not contain both keyword-based and topic-based categorization")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a comparison chart
    keyword_counts = df['main_category'].value_counts().reset_index()
    keyword_counts.columns = ['Category', 'Count']
    keyword_counts['Source'] = 'Keyword-based'
    
    topic_counts = df['topic_main_category'].value_counts().reset_index()
    topic_counts.columns = ['Category', 'Count']
    topic_counts['Source'] = 'Topic-based'
    
    combined_counts = pd.concat([keyword_counts, topic_counts])
    
    fig = px.bar(combined_counts, x='Category', y='Count', color='Source', barmode='group',
                title='Comparison of Categorization Methods',
                labels={'Category': 'Category', 'Count': 'Number of Conversations', 'Source': 'Categorization Method'})
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Conversations",
        font=dict(size=12),
        xaxis={'categoryorder':'total descending'}
    )
    
    fig.write_html(os.path.join(output_dir, "categorization_comparison.html"))
    print(f"Categorization comparison visualization saved to {os.path.join(output_dir, 'categorization_comparison.html')}")

def create_topic_to_category_sankey(df, topic_data, output_dir):
    """
    Create a Sankey diagram showing the mapping between topics and categories
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with topic modeling results
    topic_data : dict
        Dictionary with topic information
    output_dir : str
        Directory to save visualizations
    """
    if 'topic' not in df.columns or 'topic_main_category' not in df.columns:
        print("DataFrame does not contain topic and topic_main_category columns")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mapping between topics and categories
    topic_to_category = df[['topic', 'topic_main_category']].drop_duplicates()
    
    # Get topic labels
    topic_labels = {}
    for topic in df['topic'].unique():
        if topic != -1:  # Skip outlier topic
            topic_idx = str(topic)
            if topic_idx in topic_data['topic_mapping']:
                # Get first 3 words from topic
                words = topic_data['topic_mapping'][topic_idx].split(': ')[1].split(', ')[:3]
                topic_labels[topic] = f"Topic {topic}: {', '.join(words)}"
            else:
                topic_labels[topic] = f"Topic {topic}"
    
    # Create sankey data
    sources = []
    targets = []
    values = []
    labels = []
    
    # Add topic nodes
    topic_node_indices = {}
    for i, topic in enumerate(topic_to_category['topic'].unique()):
        if topic != -1:  # Skip outlier topic
            topic_node_indices[topic] = i
            labels.append(topic_labels.get(topic, f"Topic {topic}"))
    
    # Add category nodes
    category_node_indices = {}
    categories = topic_to_category['topic_main_category'].unique()
    for i, category in enumerate(categories):
        category_node_indices[category] = i + len(topic_node_indices)
        labels.append(category)
    
    # Create links
    for _, row in topic_to_category.iterrows():
        if row['topic'] != -1:  # Skip outlier topic
            sources.append(topic_node_indices[row['topic']])
            targets.append(category_node_indices[row['topic_main_category']])
            # Count number of conversations with this topic
            values.append(len(df[df['topic'] == row['topic']]))
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = "blue"
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values
        )
    )])
    
    fig.update_layout(title_text="Topic to Category Mapping", font_size=12)
    fig.write_html(os.path.join(output_dir, "topic_to_category_sankey.html"))
    print(f"Topic to category Sankey diagram saved to {os.path.join(output_dir, 'topic_to_category_sankey.html')}")

def create_advanced_visualizations(df, topic_data, output_dir):
    """
    Create various advanced visualizations for topic modeling results
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with topic modeling results
    topic_data : dict
        Dictionary with topic information
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual visualizations
    create_topic_distribution_visualization(df, output_dir)
    create_category_comparison_visualization(df, output_dir)
    create_topic_to_category_sankey(df, topic_data, output_dir)
    
    # Language distribution by topic if language information is available
    if 'is_danish' in df.columns:
        topic_language_dist = df.groupby(['topic', 'is_danish']).size().reset_index(name='Count')
        
        fig = px.bar(topic_language_dist, x='topic', y='Count', color='is_danish', barmode='group',
                    title='Language Distribution by Topic',
                    labels={'topic': 'Topic Number', 'Count': 'Number of Conversations', 'is_danish': 'Is Danish'})
        
        fig.update_layout(
            xaxis_title="Topic Number",
            yaxis_title="Number of Conversations",
            font=dict(size=12)
        )
        
        fig.write_html(os.path.join(output_dir, "language_distribution_by_topic.html"))
        print(f"Language distribution visualization saved to {os.path.join(output_dir, 'language_distribution_by_topic.html')}")
    
    # Confidence score distribution
    if 'topic_confidence' in df.columns:
        fig = px.histogram(df, x='topic_confidence', 
                          title='Distribution of Topic to Category Confidence Scores',
                          labels={'topic_confidence': 'Confidence Score', 'count': 'Number of Conversations'},
                          nbins=20)
        
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Number of Conversations",
            font=dict(size=12)
        )
        
        fig.write_html(os.path.join(output_dir, "confidence_distribution.html"))
        print(f"Confidence score distribution visualization saved to {os.path.join(output_dir, 'confidence_distribution.html')}")
    
    # Dashboard with multiple visualizations
    create_dashboard(df, topic_data, output_dir)

def create_dashboard(df, topic_data, output_dir):
    """Create an interactive dashboard with multiple views"""
    # Create HTML template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversation Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .chart { margin-bottom: 30px; }
            .chart-title { font-size: 18px; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Conversation Analysis Dashboard</h1>
            
            <div class="chart">
                <div class="chart-title">Category Distribution</div>
                <div id="category_distribution"></div>
            </div>
            
            <div class="chart">
                <div class="chart-title">Topic Distribution</div>
                <div id="topic_distribution"></div>
            </div>
            
            <div class="chart">
                <div class="chart-title">Topic to Category Mapping</div>
                <div id="topic_category_mapping"></div>
            </div>
            
            <div class="chart">
                <div class="chart-title">Method Comparison</div>
                <div id="method_comparison"></div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create category distribution chart
    category_counts = df['main_category'].value_counts()
    fig_category = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title="Category Distribution",
        labels={'x': 'Category', 'y': 'Count'}
    )
    
    # Create topic distribution chart
    topic_counts = df['topic'].value_counts()
    topic_labels = [topic_data['topic_mapping'].get(str(topic), f"Topic {topic}") for topic in topic_counts.index]
    fig_topic = px.bar(
        x=topic_labels,
        y=topic_counts.values,
        title="Topic Distribution",
        labels={'x': 'Topic', 'y': 'Count'}
    )
    
    # Create topic to category mapping using go.Sankey
    topic_category_data = []
    for topic in df['topic'].unique():
        if topic != -1:  # Skip outliers
            topic_label = topic_data['topic_mapping'].get(str(topic), f"Topic {topic}")
            category = df[df['topic'] == topic]['topic_main_category'].iloc[0]
            count = len(df[df['topic'] == topic])
            topic_category_data.append({
                'source': topic_label,
                'target': category,
                'value': count
            })
    
    # Create Sankey diagram
    fig_mapping = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = list(set([d['source'] for d in topic_category_data] + [d['target'] for d in topic_category_data])),
            color = "blue"
        ),
        link = dict(
            source = [list(set([d['source'] for d in topic_category_data] + [d['target'] for d in topic_category_data])).index(d['source']) for d in topic_category_data],
            target = [list(set([d['source'] for d in topic_category_data] + [d['target'] for d in topic_category_data])).index(d['target']) for d in topic_category_data],
            value = [d['value'] for d in topic_category_data]
        )
    )])
    
    fig_mapping.update_layout(title_text="Topic to Category Mapping")
    
    # Create method comparison chart
    method_comparison = pd.crosstab(df['main_category'], df['topic_main_category'])
    fig_comparison = px.imshow(
        method_comparison,
        title="Keyword vs Topic-based Categorization Comparison",
        labels={'x': 'Topic-based Category', 'y': 'Keyword-based Category'}
    )
    
    # Save the dashboard
    dashboard_path = os.path.join(output_dir, "dashboard.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save the plots
    plotly.io.write_html(fig_category, os.path.join(output_dir, "category_distribution.html"))
    plotly.io.write_html(fig_topic, os.path.join(output_dir, "topic_distribution.html"))
    plotly.io.write_html(fig_mapping, os.path.join(output_dir, "topic_category_mapping.html"))
    plotly.io.write_html(fig_comparison, os.path.join(output_dir, "method_comparison.html"))
    
    print(f"Dashboard saved to: {dashboard_path}")

def visualize_results(df, topic_data, output_dir):
    """
    Create all visualizations for topic modeling results
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with topic modeling results
    topic_data : dict
        Dictionary with topic information
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create advanced visualizations
    create_advanced_visualizations(df, topic_data, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    import topic_modeling
    import categorization
    
    # Define file paths
    cleaned_csv = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    embedding_file = os.path.join("..", "data", "embeddings", "conversation_embeddings.npy")
    output_dir = os.path.join("..", "data", "processed", "topic_modeling")
    
    # Load topic modeling results
    if os.path.exists(os.path.join(output_dir, "topic_modeled_conversations.csv")):
        df = pd.read_csv(os.path.join(output_dir, "topic_modeled_conversations.csv"))
        with open(os.path.join(output_dir, "topic_data.json"), 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
            
        # Create visualizations
        visualize_results(df, topic_data, os.path.join(output_dir, "visualizations"))
    else:
        print("Topic modeling results not found. Run topic_modeling.py first.") 