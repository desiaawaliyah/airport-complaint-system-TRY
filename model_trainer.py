import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from text_processor import preprocess_text
from sample_data import get_sample_data_with_labels

def train_model():
    """
    Train a complaint classification model
    
    Returns:
        model: Trained classification model
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted label encoder
    """
    # Get sample data with labels for training
    # In a real implementation, you would load your own training data here
    train_data = get_sample_data_with_labels()
    
    # Convert to DataFrame
    df = pd.DataFrame(train_data, columns=['text', 'category'])
    
    # Preprocess the text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Encode the categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['category_encoded'],
        test_size=0.2,
        random_state=42
    )
    
    # Create and train the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate the model
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, vectorizer, label_encoder

def save_model(model, vectorizer, label_encoder, model_path='model.pkl', vectorizer_path='vectorizer.pkl', encoder_path='label_encoder.pkl'):
    """
    Save the trained model and vectorizer
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        label_encoder: Fitted label encoder
        model_path: Path to save the model
        vectorizer_path: Path to save the vectorizer
        encoder_path: Path to save the label encoder
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl', encoder_path='label_encoder.pkl'):
    """
    Load the trained model and vectorizer or train a new one if not found
    
    Args:
        model_path: Path to the saved model
        vectorizer_path: Path to the saved vectorizer
        encoder_path: Path to the saved label encoder
        
    Returns:
        model: Loaded or trained model
        vectorizer: Loaded or trained vectorizer
        label_encoder: Loaded or trained label encoder
    """
    # Check if the model files exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path):
        # Load the model and vectorizer
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        # Train a new model
        model, vectorizer, label_encoder = train_model()
        save_model(model, vectorizer, label_encoder)
    
    return model, vectorizer, label_encoder

def predict_category(processed_text, model, vectorizer, label_encoder):
    """
    Predict the category of a preprocessed complaint text
    
    Args:
        processed_text (str): Preprocessed complaint text
        model: Trained classification model
        vectorizer: Fitted vectorizer
        label_encoder: Fitted label encoder
        
    Returns:
        str: Predicted category
        float: Confidence score (percentage)
    """
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Predict the category
    prediction = model.predict(text_vectorized)[0]
    
    # Get the confidence score
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = probabilities[prediction] * 100
    
    # Convert prediction back to category name
    category = label_encoder.inverse_transform([prediction])[0]
    
    return category, confidence
