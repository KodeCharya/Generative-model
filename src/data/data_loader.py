import pandas as pd
import re

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        if self.file_path.endswith('.txt'):
            with open(self.file_path, 'r') as file:
                data = file.read()
        elif self.file_path.endswith('.csv'):
            data = pd.read_csv(self.file_path)
        else:
            raise ValueError("Unsupported file format")
        return data

    def preprocess(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text