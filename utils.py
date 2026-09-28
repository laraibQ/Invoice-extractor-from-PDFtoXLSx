import re
import numpy as np
import cv2
import pandas as pd
from typing import List, Dict, Any, Optional, Set

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for better table detection and OCR.
    
    Args:
        img: Input image as numpy array (BGR format)
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive thresholding to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opening

def clean_text(text: Any) -> str:
    """
    Clean and normalize text from OCR or PDF extraction.
    
    Args:
        text: Input text (can be any type)
        
    Returns:
        Cleaned text as string
    """
    if text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove unusual characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Normalize dashes and quotes
    text = text.replace('–', '-').replace('—', '-').replace('"', '"').replace('"', '"')
    
    # Remove common OCR errors
    text = text.replace('|', 'I').replace('l', 'I')
    
    return text.strip()

def find_header_row(table: List[List[str]], header_keywords: List[str]) -> int:
    """
    Find the row index that is most likely to be the header row.
    
    Args:
        table: Table as a list of rows
        header_keywords: List of keywords that commonly appear in headers
        
    Returns:
        Index of the header row, or -1 if not found
    """
    if not table:
        return -1
    
    max_score = -1
    header_idx = -1
    
    for i, row in enumerate(table):
        if not row:
            continue
        
        # Calculate a score based on how many header keywords are in this row
        row_text = ' '.join(str(cell).lower() for cell in row if cell)
        score = sum(1 for keyword in header_keywords if keyword in row_text)
        
        # Give a higher weight to rows with multiple keywords
        if score > max_score:
            max_score = score
            header_idx = i
    
    # If score is too low, it might not be a header
    return header_idx if max_score >= 2 else -1

def normalize_currency(text: str) -> str:
    """
    Normalize currency values for consistent formatting.
    
    Args:
        text: Input text containing currency values
        
    Returns:
        Normalized text
    """
    # Handle common currency symbols and formats
    # Replace currency symbols with standard format
    text = re.sub(r'[$£€]', '', text)
    
    # Handle comma as thousands separator
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Ensure decimal point is '.'
    text = text.replace(',', '.')
    
    return text

def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply post-processing to a DataFrame to clean and normalize data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame
    """
    if df.empty:
        return df
    
    # 1. Forward fill NaN values to handle merged cells
    df = df.fillna(method='ffill', axis=0)
    
    # 2. Identify and clean numeric columns
    for col in df.columns:
        # Check if column values have currency/numeric patterns
        num_currency_pattern = sum(1 for val in df[col] if isinstance(val, str) and 
                                 re.search(r'[$£€]?\s?\d+[\.,]\d+', val))
        
        if num_currency_pattern > len(df) * 0.5:  # More than 50% of values are currency
            # Clean currency values
            df[col] = df[col].apply(lambda x: normalize_currency(str(x)) if isinstance(x, (str, int, float)) else x)
    
    # 3. Remove rows that appear to be headers duplicated in the data
    # First, get column names as lowercase set for comparison
    col_names_lower = {str(col).lower() for col in df.columns}
    
    # Check each row to see if it might be a header
    header_rows = []
    for idx, row in df.iterrows():
        row_values = [str(val).lower() for val in row.values if not pd.isna(val)]
        
        # Count how many row values match column names
        matches = sum(1 for val in row_values if any(col.find(val) >= 0 for col in col_names_lower))
        
        # If more than half the values match column names, likely a header
        if matches > len(row_values) * 0.5:
            header_rows.append(idx)
    
    # Drop identified header rows
    if header_rows:
        df = df.drop(header_rows)
    
    # 4. Check for empty columns and remove them
    df = df.dropna(axis=1, how='all')
    
    # 5. Check for columns with identical values and merge them
    duplicate_cols = []
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                duplicate_cols.append(df.columns[j])
    
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)
    
    # 6. Reset index
    df = df.reset_index(drop=True)
    
    return df
