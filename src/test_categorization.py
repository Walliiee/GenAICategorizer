import unittest
import pandas as pd
import categorization

class TestCategorization(unittest.TestCase):
    
    def test_is_voice_conversation(self):
        # Test voice detection
        self.assertTrue(categorization.is_voice_conversation("This is a transcript of a voice conversation"))
        self.assertTrue(categorization.is_voice_conversation("Audio file contains speech"))
        self.assertFalse(categorization.is_voice_conversation("Regular text conversation without voice"))
    
    def test_is_danish_text(self):
        # Test Danish detection
        self.assertTrue(categorization.is_danish_text("Hvordan kan jeg hjælpe dig med dette problem?"))
        self.assertTrue(categorization.is_danish_text("Jeg vil gerne vide hvordan man laver en god opslag på LinkedIn"))
        self.assertFalse(categorization.is_danish_text("This is English text with no Danish words"))
    
    def test_get_category_scores(self):
        # Test category scoring
        text = "How do I write code to debug this Python error in my function?"
        scores = categorization.get_category_scores(text)
        
        # Check that Code Development has high score
        self.assertTrue(scores['Code Development']['total_score'] > 0)
        
        # Check that Bug Fixing has high subcategory score within Code Development
        self.assertTrue(scores['Code Development']['sub_scores']['Bug Fixing'] > 0)
    
    def test_assign_categories(self):
        # Test category assignment
        text = "How do I write code to debug this Python error in my function?"
        result = categorization.assign_categories(text)
        
        # Check main category is Code Development
        self.assertEqual(result['main_category'], 'Code Development')
        
        # Check subcategory is Bug Fixing
        self.assertEqual(result['subcategory'], 'Bug Fixing')
        
        # Check not voice
        self.assertFalse(result['is_voice'])
    
    def test_apply_keyword_categorization(self):
        # Create test dataframe
        df = pd.DataFrame({
            'text': [
                "How do I write code to debug this Python error?",
                "Can you help me draft an email to my client?",
                "Explain the concept of machine learning",
                "This is a transcript of voice conversation"
            ]
        })
        
        # Apply categorization
        result_df = categorization.apply_keyword_categorization(df)
        
        # Check dataframe has required columns
        self.assertTrue('main_category' in result_df.columns)
        self.assertTrue('subcategory' in result_df.columns)
        self.assertTrue('is_voice' in result_df.columns)
        self.assertTrue('complexity' in result_df.columns)
        
        # Check categorization results
        self.assertEqual(result_df.loc[0, 'main_category'], 'Code Development')
        self.assertEqual(result_df.loc[1, 'main_category'], 'Writing Assistance')
        self.assertEqual(result_df.loc[2, 'main_category'], 'Learning/Education')
        self.assertTrue(result_df.loc[3, 'is_voice'])

if __name__ == '__main__':
    unittest.main() 