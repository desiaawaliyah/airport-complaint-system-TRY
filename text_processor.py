import re
import os

# Custom simple tokenizer
def simple_word_tokenize(text):
    """A simplified tokenizer that just splits on whitespace"""
    return text.split()

# Simple stemmer function
def simple_stem(word):
    """
    A very basic stemmer that just removes common suffixes
    This is not as good as proper stemmers but avoids NLTK dependencies
    """
    suffixes = ['ing', 'ly', 'ed', 'es', 's', 'er', 'est', 'ment', 'ness']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

# Indonesian and English stopwords
STOPWORDS = {
    # English stopwords
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    
    # Indonesian stopwords
    'yang', 'dan', 'di', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam', 'akan', 'pada', 'juga',
    'saya', 'ke', 'karena', 'tersebut', 'bisa', 'ada', 'mereka', 'lebih', 'telah', 'saat', 'itu',
    'atau', 'hanya', 'kita', 'secara', 'oleh', 'seperti', 'tapi', 'sebagai', 'para', 'dapat', 'dia',
    'bahwa', 'setelah', 'harus', 'ketika', 'belum', 'lagi', 'kami', 'sudah', 'anda', 'lalu', 'satu',
    'semakin', 'hingga', 'kalau', 'menurut', 'tentang', 'sekitar', 'sebuah', 'antara', 'selama', 'nya',
    'ku', 'mu', 'kamu'
}

def preprocess_text(text):
    """
    Preprocess text for Indonesian language complaints
    
    Args:
        text (str): Raw complaint text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize with simple tokenizer to avoid NLTK issues
    tokens = simple_word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Remove short tokens (less than 3 characters)
    tokens = [token for token in tokens if len(token) >= 3]
    
    # Stemming with simple stemmer to avoid NLTK dependencies
    # Note: This is a very basic stemmer. In a production system, 
    # consider using Sastrawi or another Indonesian-focused stemmer.
    tokens = [simple_stem(token) for token in tokens]
    
    # Join tokens back to text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
