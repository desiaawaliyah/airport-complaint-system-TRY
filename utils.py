import pandas as pd

def search_complaints(df, query):
    """
    Search for complaints containing the query string
    
    Args:
        df (pd.DataFrame): DataFrame containing complaints
        query (str): Search query
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing matching complaints
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Search in the text column
    mask = df['text'].str.lower().str.contains(query, na=False)
    
    # Return matching rows
    return df[mask]

def export_to_csv(df, filename='komplain_terklasifikasi.csv'):
    """
    Export the complaints DataFrame to a CSV file
    
    Args:
        df (pd.DataFrame): DataFrame to export
        filename (str): Output filename
        
    Returns:
        str: Path to the saved file
    """
    if df.empty:
        return None
    
    df.to_csv(filename, index=False)
    return filename
