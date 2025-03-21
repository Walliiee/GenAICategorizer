import unittest
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
from topic_modeling import load_or_create_embeddings, create_topic_model, analyze_topics

class TestTopicModeling(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test dataframe
        self.df = pd.DataFrame({
            'text': [
                "How do I write code to debug this Python error?",
                "Can you help me debug this JavaScript code?",
                "I need to fix a bug in my Python program.",
                "What's wrong with my code? I'm getting an error.",
                "Can you help me draft an email to my client?",
                "Write a professional email for a job application.",
                "Help me draft a cover letter for this position.",
                "Can you write a thank you email after an interview?",
                "Explain the concept of machine learning",
                "What is deep learning and how does it differ from ML?",
                "Can you teach me about neural networks?",
                "What is the difference between supervised and unsupervised learning?"
            ]
        })
        
        # Save test dataframe
        self.csv_file = os.path.join(self.test_dir, "test_conversations.csv")
        self.df.to_csv(self.csv_file, index=False)
        
        # Create test embeddings
        self.embedding_file = os.path.join(self.test_dir, "test_embeddings.npy")
        self.embeddings = np.random.rand(len(self.df), 384)  # Simulate embeddings
        np.save(self.embedding_file, self.embeddings)
        
        # Output directory for topic modeling
        self.output_dir = os.path.join(self.test_dir, "topic_modeling_output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_load_or_create_embeddings(self):
        # Test loading existing embeddings
        df, embeddings = load_or_create_embeddings(self.csv_file, self.embedding_file)
        
        # Check that the correct dataframe and embeddings are loaded
        self.assertEqual(len(df), len(self.df))
        self.assertTrue(np.array_equal(embeddings, self.embeddings))
    
    @unittest.skip("Skipping topic model creation as it requires downloading models")
    def test_create_topic_model(self):
        # This test would be more involved and require downloading models
        # For integration testing rather than unit testing
        topic_model, topics, probs = create_topic_model(
            self.df, self.embeddings, min_topic_size=2, nr_topics=3
        )
        
        # Check that topics were assigned
        self.assertEqual(len(topics), len(self.df))
        
        # Check that we have the expected number of topics
        unique_topics = set(topics)
        self.assertTrue(len(unique_topics) <= 4)  # 3 topics + possibly outlier topic (-1)
    
    @unittest.skip("Skipping topic analysis as it requires topic model creation")
    def test_analyze_topics(self):
        # This test depends on create_topic_model
        # For integration testing rather than unit testing
        pass

if __name__ == '__main__':
    unittest.main() 