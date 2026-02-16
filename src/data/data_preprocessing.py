import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_comment(comment):
    try:
        # Convert to lowercase
        comment = comment.lower()
        
        # Remove trailing and leading whitespaces
        comment = comment.strip()
        
        # Remove newline characters
        comment = re.sub(r'\n', '', comment)
        
        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        # Remove extra whitespaces
        comment = re.sub(r'\s+', ' ', comment)
        
        # Remove URLs
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
        
        # Remove HTML tags
        comment = re.sub(r'<.*?>', '', comment)
        
        # Remove mentions (@username)
        comment = re.sub(r'@\w+', '', comment)
        
        # Remove hashtags (but keep the text)
        comment = re.sub(r'#(\w+)', r'\1', comment)
        
        # Remove numbers (optional - can be kept for context)
        # comment = re.sub(r'\d+', '', comment)
        
        # Handle contractions
        contraction_dict = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it's": "it is",
            "let's": "let us",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Replace contractions
        for contraction, expansion in contraction_dict.items():
            comment = comment.replace(contraction, expansion)
        
        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet', 'very', 'too', 'won'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word, pos='v') for word in comment.split()])
        
        # Remove very short words (optional)
        comment = ' '.join([word for word in comment.split() if len(word) > 1])
        
        # Final cleanup of extra spaces
        comment = re.sub(r'\s+', ' ', comment).strip()
        
        return comment
    except Exception as e:
        logger.error(f"Error during comment preprocessing: {e}")
        raise

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, create_raw_subfolder: bool = True) -> None:
    """Save the train and test datasets.
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        data_path: Path to save data
        create_raw_subfolder: If True, creates a 'interim' subfolder
    """
    try:
        if create_raw_subfolder:
            interim_data_path = os.path.join(data_path, 'interim')
        else:
            interim_data_path = data_path

        # Create the directory if it does not exist
        os.makedirs(interim_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug('Train and test data saved to %s', interim_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Fetch the data from artifacts/data/raw
        train_data = pd.read_csv('artifacts/data/train.csv')
        test_data = pd.read_csv('artifacts/data/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data (make sure normalize_text function is defined)
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data to artifacts/data/processed
        save_data(train_processed_data, test_processed_data, data_path='artifacts/interim', create_raw_subfolder=False)
        logger.debug('Processed data saved to artifacts/interim')

    except FileNotFoundError as e:
        logger.error('Data files not found: %s', e)
        logger.error('Make sure data_ingestion stage has run and files exist in artifacts/data')
        raise
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()  # Add this line to actually call the function