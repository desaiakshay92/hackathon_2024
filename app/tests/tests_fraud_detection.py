import unittest
import os
# from fraud_detection import load_data, preprocess_data, train_model, preprocess_input, predict_label
from io import StringIO
from app.models.fraud_detection import load_data, preprocess_data, train_model, preprocess_input, cv ,predict_label

class TestFraudDetection(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.data = StringIO("""1\tThis is a normal call.
                                0\tThis seems like a fraud call.
                                1\tJust another normal conversation.
                                0\tSuspicious activity detected!""")
        self.df = load_data(self.data)
        self.test_file = 'data/fraud_call.file'
    
    def test_file_exists(self):
        # Check if the file exists
        self.assertTrue(os.path.exists(self.test_file), "File does not exist.")
    
    def test_file_not_empty(self):
        # Check if the file is not empty
        self.assertGreater(os.path.getsize(self.test_file), 0, "File is empty.")

    def test_preprocess_data(self):
        # Test preprocessing of data
        corpus = preprocess_data(self.df)
        expected_corpus = [
            'normal call',
            'seems like fraud call',
            'another normal conversation',
            'suspicious activity detected'
        ]
        self.assertEqual(corpus, expected_corpus)


    def test_empty_input(self):
        # Test empty input
        corpus = preprocess_data(self.df)
        model = train_model(corpus, self.df['label'])
        prediction = predict_label("", model)
        # Assuming empty input would result in a prediction, though it depends on preprocessing
        self.assertIn(prediction, ['fraud', 'normal call'])

if __name__ == '__main__':
    unittest.main()