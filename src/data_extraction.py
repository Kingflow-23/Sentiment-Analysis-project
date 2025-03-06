import pandas as pd

def extraction(path):
    """
    This function will extract the data from a CSV file and handle errors like missing values and wrong formats.
    It will also keep only the 'content' and 'score' columns.
    """

    try:
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Check if the necessary columns exist in the dataframe
        if 'content' in df.columns and 'score' in df.columns:
            # Keep only the 'content' and 'score' columns
            df = df[['content', 'score']]
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Optionally, handle missing values (e.g., drop rows with missing 'content' or 'score' values)
        df = df.dropna(subset=['content', 'score'])
        
        return df
    
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        return None
