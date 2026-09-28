#!/usr/bin/env python3
import os
import re
import sys
import io
import logging
import argparse
import tempfile
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pdfplumber
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Windows-compatible function to find executable path
def find_executable_path(executable_name):
    """Find executable path in a Windows-compatible way"""
    # For Windows, check if the executable is in the PATH
    if sys.platform == 'win32':
        # Check common install locations for tesseract
        if executable_name == 'tesseract':
            common_paths = [
                r"C:\Users\User\Tesseract-OCR\tesseract.exe",
                r"C:\Users\User\Tesseract-OCR\tesseract.exe",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
            # If not found in common paths, try to find in PATH
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, executable_name + '.exe')
                if os.path.isfile(exe_file):
                    return exe_file
            return None
    else:
        # For Unix/Linux/Mac
        try:
            path = subprocess.check_output(['which', executable_name]).decode().strip()
            return path if path else None
        except:
            return None

# Helper functions
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess an image for better table detection and OCR."""
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
    """Clean and normalize text from OCR or PDF extraction."""
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
    """Find the row index that is most likely to be the header row."""
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

def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply post-processing to a DataFrame to clean and normalize data."""
    if df.empty:
        return df
    
    # 1. Forward fill NaN values to handle merged cells
    df = df.fillna(method='ffill', axis=0)
    
    # 2. Clean column names
    df.columns = [clean_text(col) for col in df.columns]
    
    # 3. Drop completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # 4. Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 5. Handle multi-line text in cells
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    
    # 6. Trim whitespace in text cells
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

class InvoiceProcessor:
    """
    Main class to extract invoice tables from PDFs and convert to CSV.
    """
    
    def __init__(self, tesseract_path=None, debug=False):
        """Initialize the processor."""
        self.debug = debug
        
        # Set tesseract path if provided or search for it
        if not tesseract_path:
            tesseract_path = find_executable_path('tesseract')
            if tesseract_path:
                logger.info(f"Found tesseract at: {tesseract_path}")
            else:
                logger.warning("Tesseract not found. OCR functionality may not work.")
                logger.warning("Please install Tesseract OCR and add it to your PATH, or specify the path manually.")
                # For Windows users, provide more info
                if sys.platform == 'win32':
                    logger.warning("On Windows, download from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Common keywords found in invoice table headers
        self.header_keywords = [
            'description', 'desc', 'item', 'product', 'service', 'article',
            'quantity', 'qty', 'quant', 'units', 'pcs',
            'price', 'rate', 'unit price', 'unit cost', 'amount',
            'total', 'subtotal', 'net', 'gross', 'sum',
            'vat', 'tax', 'gst', 'hst', 'pst',
            'discount', 'code', 'sku', 'ref', 'reference',
            'date', 'invoice', 'order', 'delivery', 'payment'
        ]
    
    def process_pdf(self, pdf_path, output_path=None):
        """
        Process a PDF invoice and extract tables to CSV.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the CSV (if None, uses pdf_path with .csv extension)
            
        Returns:
            Success status (bool)
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.splitext(pdf_path)[0] + '.csv'
        
        logger.info(f"Processing {pdf_path}")
        logger.info(f"Output will be saved to {output_path}")
        
        try:
            # Try pdfplumber first
            logger.info("Trying extraction with pdfplumber...")
            tables = self._extract_with_pdfplumber(pdf_path)
            
            # If pdfplumber fails, try OCR
            if not tables or all(t.empty for t in tables if t is not None):
                logger.info("Pdfplumber extraction yielded no results, trying OCR...")
                tables = self._extract_with_ocr(pdf_path)
            
            # If OCR fails, try text-based extraction
            if not tables or all(t.empty for t in tables if t is not None):
                logger.info("OCR extraction yielded no results, trying text-based extraction...")
                tables = self._extract_from_text(pdf_path)
            
            # If all methods fail, return failure
            if not tables or all(t.empty for t in tables if t is not None):
                logger.error("No tables found in the invoice using any method")
                return False
            
            # Select the best table (usually the largest one with invoice-like columns)
            best_table = self._select_best_table(tables)
            
            if best_table is None or best_table.empty:
                logger.error("No valid table found in the invoice")
                return False
            
            # Post-process the table
            final_table = postprocess_dataframe(best_table)
            
            # Save to CSV
            final_table.to_csv(output_path, index=False)
            logger.info(f"Successfully saved CSV to {output_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract tables using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_num + 1} with pdfplumber")
                    
                    # Try different extraction settings
                    extraction_settings = [
                        {},  # Default settings
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                    ]
                    
                    for settings in extraction_settings:
                        try:
                            page_tables = page.extract_tables(table_settings=settings)
                            for table in page_tables:
                                if table and len(table) > 1:  # Skip empty tables
                                    # Identify header row
                                    header_row_idx = find_header_row(table, self.header_keywords)
                                    
                                    if header_row_idx >= 0:
                                        header = table[header_row_idx]
                                        data = table[header_row_idx+1:]
                                    else:
                                        header = table[0]
                                        data = table[1:]
                                    
                                    # Convert to DataFrame
                                    df = pd.DataFrame(data, columns=header)
                                    df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
                                    df = df.replace(r'^\s*$', pd.NA, regex=True)
                                    df = df.dropna(how='all')
                                    
                                    if not df.empty and len(df.columns) >= 2:
                                        tables.append(df)
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error with extraction settings: {str(e)}")
                            continue
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {str(e)}")
        
        return tables
    
    def _extract_with_ocr(self, pdf_path):
        """Extract tables using OCR."""
        tables = []
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            for i, img in enumerate(images):
                logger.debug(f"Processing page {i+1} with OCR")
                
                # Convert to OpenCV format
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Try to OCR the whole page
                ocr_text = pytesseract.image_to_string(img)
                
                # Try to parse the text as a table
                lines = ocr_text.split('\n')
                potential_table = []
                
                for line in lines:
                    if line.strip():
                        # Split by multiple spaces or tabs
                        row = re.split(r'\s{2,}|\t', line)
                        row = [cell.strip() for cell in row if cell.strip()]
                        if row:
                            potential_table.append(row)
                
                if potential_table:
                    # Find potential header based on keywords
                    header_idx = find_header_row(potential_table, self.header_keywords)
                    
                    if header_idx >= 0:
                        # Use identified header row
                        header = potential_table[header_idx]
                        data_rows = potential_table[header_idx+1:]
                    else:
                        # Use first row as header
                        header = potential_table[0]
                        data_rows = potential_table[1:]
                    
                    # Ensure all rows have same number of columns
                    max_cols = max(len(row) for row in data_rows) if data_rows else len(header)
                    max_cols = max(max_cols, len(header))
                    
                    # Pad header if needed
                    if len(header) < max_cols:
                        header = header + ['Column' + str(i+1) for i in range(len(header), max_cols)]
                    
                    # Pad data rows
                    padded_rows = []
                    for row in data_rows:
                        if len(row) < max_cols:
                            row = row + [''] * (max_cols - len(row))
                        padded_rows.append(row[:max_cols])
                    
                    # Create DataFrame
                    if padded_rows:
                        df = pd.DataFrame(padded_rows, columns=header[:max_cols])
                        df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
                        df = df.replace(r'^\s*$', pd.NA, regex=True)
                        df = df.dropna(how='all')
                        
                        if not df.empty and len(df.columns) >= 2:
                            tables.append(df)
        
        except Exception as e:
            logger.error(f"OCR extraction error: {str(e)}")
        
        return tables
    
    def _extract_from_text(self, pdf_path):
        """Extract tables from text when other methods fail."""
        tables = []
        try:
            # Extract raw text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n\n"
            
            # Split into lines
            lines = all_text.split('\n')
            
            # Find potential header lines
            header_line_indices = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                keyword_count = sum(1 for keyword in self.header_keywords if keyword in line_lower)
                if keyword_count >= 2:
                    header_line_indices.append((i, keyword_count))
            
            # Sort by keyword count (highest first)
            header_line_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Try each potential header
            for header_idx, _ in header_line_indices:
                header_line = lines[header_idx]
                
                # Split the header line
                header_cols = re.split(r'\s{2,}', header_line.strip())
                header_cols = [col.strip() for col in header_cols if col.strip()]
                
                if len(header_cols) >= 2:
                    # Extract data rows
                    data_rows = []
                    
                    # Process lines after the header
                    for i in range(header_idx + 1, len(lines)):
                        line = lines[i].strip()
                        if not line:
                            continue
                        
                        # Check if line has numbers (likely data)
                        if re.search(r'\d', line):
                            # Split in the same way as header
                            cols = re.split(r'\s{2,}', line)
                            cols = [col.strip() for col in cols if col.strip()]
                            
                            if len(cols) >= 2:
                                data_rows.append(cols)
                                
                        # Break if we hit a likely end of table
                        if re.search(r'\b(total|subtotal|sum|amount)\b', line.lower()):
                            # Add this line as it may contain valuable data
                            cols = re.split(r'\s{2,}', line)
                            cols = [col.strip() for col in cols if col.strip()]
                            if len(cols) >= 2:
                                data_rows.append(cols)
                            break
                    
                    # Create DataFrame if we have data
                    if data_rows:
                        # Normalize lengths
                        max_cols = max(len(row) for row in data_rows)
                        max_cols = max(max_cols, len(header_cols))
                        
                        # Pad header if needed
                        if len(header_cols) < max_cols:
                            header_cols = header_cols + ['Column' + str(i+1) for i in range(len(header_cols), max_cols)]
                        
                        # Pad data rows
                        padded_rows = []
                        for row in data_rows:
                            if len(row) < max_cols:
                                row = row + [''] * (max_cols - len(row))
                            padded_rows.append(row[:max_cols])
                        
                        # Create DataFrame
                        df = pd.DataFrame(padded_rows, columns=header_cols[:max_cols])
                        df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
                        df = df.replace(r'^\s*$', pd.NA, regex=True)
                        df = df.dropna(how='all')
                        
                        if not df.empty and len(df.columns) >= 2:
                            tables.append(df)
                            break  # Found a good table, stop looking
                            
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
        
        return tables
    
    def _select_best_table(self, tables):
        """Select the best table from a list of candidates."""
        if not tables:
            return None
        
        # Score tables based on quality criteria
        table_scores = []
        for table in tables:
            if table is None or table.empty:
                continue
                
            score = 0
            
            # 1. Number of columns (more is better, up to a point)
            num_cols = len(table.columns)
            if num_cols >= 4:
                score += 3
            elif num_cols >= 3:
                score += 2
            elif num_cols >= 2:
                score += 1
            
            # 2. Number of rows (more is better)
            num_rows = len(table)
            if num_rows >= 5:
                score += 3
            elif num_rows >= 3:
                score += 2
            elif num_rows >= 1:
                score += 1
            
            # 3. Presence of key column headers
            headers = [str(col).lower() for col in table.columns]
            header_score = sum(1 for keyword in self.header_keywords 
                               if any(keyword in header for header in headers))
            score += min(header_score, 5)
            
            # 4. Presence of numerical data (likely invoice amounts)
            num_numeric_cells = 0
            for col in table.columns:
                for val in table[col]:
                    if isinstance(val, str) and re.search(r'\d+\.?\d*', val):
                        num_numeric_cells += 1
                    elif isinstance(val, (int, float)):
                        num_numeric_cells += 1
            
            if num_numeric_cells >= 10:
                score += 4
            elif num_numeric_cells >= 5:
                score += 2
            elif num_numeric_cells >= 1:
                score += 1
            
            table_scores.append((table, score))
        
        # Sort by score (highest first)
        table_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return highest scoring table
        return table_scores[0][0] if table_scores else None

def main():
    parser = argparse.ArgumentParser(description='Extract tables from invoice PDFs')
    parser.add_argument('pdf_path', help='Path to the invoice PDF file or directory containing PDFs')
    parser.add_argument('-o', '--output', help='Output CSV file path or directory (if input is directory)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    
    # Parse args (but handle args with spaces correctly)
    if len(sys.argv) > 1:
        # Force quotes around file paths with spaces
        args = parser.parse_args()
    else:
        parser.print_help()
        sys.exit(1)
    
    # Initialize the processor
    processor = InvoiceProcessor(debug=args.debug)
    
    # Check if input is directory or file
    if os.path.isdir(args.pdf_path):
        # Process all PDFs in directory
        pdf_files = [f for f in os.listdir(args.pdf_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in directory: {args.pdf_path}")
            return 1
        
        # Create output directory if provided
        output_dir = args.output if args.output else args.pdf_path
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each PDF
        success_count = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.pdf_path, pdf_file)
            output_name = os.path.splitext(pdf_file)[0] + '.csv'
            output_path = os.path.join(output_dir, output_name)
            
            logger.info(f"Processing {pdf_file}...")
            if processor.process_pdf(pdf_path, output_path):
                success_count += 1
        
        logger.info(f"Processed {success_count} of {len(pdf_files)} files successfully")
        return 0 if success_count > 0 else 1
    
    else:
        # Process single PDF file
        if not os.path.exists(args.pdf_path):
            logger.error(f"File not found: {args.pdf_path}")
            return 1
        
        # Set output path
        output_path = args.output if args.output else os.path.splitext(args.pdf_path)[0] + '.csv'
        
        # Process the PDF
        success = processor.process_pdf(args.pdf_path, output_path)
        return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())