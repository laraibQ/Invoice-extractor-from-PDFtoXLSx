import os
import re
import io
import logging
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pdfplumber
import warnings
import subprocess
from typing import List, Dict, Tuple, Optional, Union, Any
from table_detector import TableDetector
from utils import preprocess_image, clean_text, postprocess_dataframe, find_header_row

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Locate system executables (tesseract, etc.)
def find_executable_path(executable_name):
    import shutil
    path = shutil.which(executable_name)
    if path:
        return path
    try:
        path = subprocess.check_output(["which", executable_name], stderr=subprocess.DEVNULL).decode().strip()
        return path if path else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

class InvoiceTableExtractor:
    """
    Advanced invoice table extractor that can detect and extract tables from PDF invoices
    using multiple extraction methods with fallback strategies.
    """
    
    def __init__(self, 
                 tesseract_path: Optional[str] = None, 
                 poppler_path: Optional[str] = None,
                 debug: bool = False,
                 min_confidence: float = 60.0):
        """
        Initialize the invoice table extractor.
        
        Args:
            tesseract_path: Path to tesseract executable
            poppler_path: Path to poppler binaries
            debug: Enable debug mode with verbose logging and visualization
            min_confidence: Minimum confidence score for table detection (0-100)
        """
        self.debug = debug
        self.min_confidence = min_confidence
        
        # Set tesseract path - try to find it if not provided
        if not tesseract_path:
            tesseract_path = find_executable_path('tesseract')
            logger.info(f"Found tesseract at: {tesseract_path}")
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Set poppler path - not needed in Linux environments as it's in PATH
        self.poppler_path = None
        
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
        
        # Initialize table detector
        self.table_detector = TableDetector(debug=debug)
        
        logger.info("InvoiceTableExtractor initialized")
        
    def extract_from_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to extract tables from PDF invoice.
        Tries multiple extraction methods and returns the best result.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the extracted CSV
            
        Returns:
            pandas DataFrame containing the extracted table
        """
        logger.info(f"Processing {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try multiple extraction methods and use the best result
        extraction_methods = [
            ("pdfplumber_extract", self.extract_with_pdfplumber),
            ("ocr_extract", self.extract_with_ocr),
            ("hybrid_extract", self.extract_with_hybrid_approach),
            ("text_based_extract", self.extract_from_text)
        ]
        
        best_table = None
        best_score = 0
        
        for method_name, method in extraction_methods:
            logger.info(f"Trying extraction method: {method_name}")
            try:
                tables = method(pdf_path)
                if tables:
                    # Evaluate quality of extracted tables
                    for i, table in enumerate(tables):
                        if table is not None and not table.empty:
                            score = self._evaluate_table_quality(table)
                            logger.debug(f"{method_name} table {i} quality score: {score}")
                            
                            if score > best_score:
                                best_score = score
                                best_table = table
                                logger.info(f"Found better table with {method_name}, score: {score}")
            except Exception as e:
                logger.error(f"Error with {method_name}: {str(e)}")
        
        if best_table is None:
            logger.warning("No valid table found in the invoice")
            return pd.DataFrame()
        
        # Post-process the table to clean it up
        final_table = self._post_process_table(best_table)
        
        # Save to CSV if output path is provided
        if output_path and final_table is not None and not final_table.empty:
            self._save_to_csv(final_table, output_path)
        
        return final_table

    def extract_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using pdfplumber with multiple table settings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing the extracted tables
        """
        logger.info("Extracting with pdfplumber")
        extracted_tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Handle multi-page PDFs
                for page_num, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_num + 1}")
                    
                    # Try different table extraction settings
                    extraction_settings = [
                        {},  # Default settings
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict", "snap_tolerance": 3},
                        {"vertical_strategy": "explicit", "horizontal_strategy": "explicit", 
                         "explicit_vertical_lines": page.curves + page.edges, 
                         "explicit_horizontal_lines": page.curves + page.edges}
                    ]
                    
                    page_tables = []
                    
                    for settings in extraction_settings:
                        try:
                            tables = page.extract_tables(table_settings=settings)
                            if tables:
                                for table in tables:
                                    if table and len(table) > 1:  # Skip empty tables
                                        # Check if this looks like a header row
                                        header_row_idx = find_header_row(table, self.header_keywords)
                                        
                                        # If we found a header row, use it as the header
                                        if header_row_idx >= 0:
                                            header = table[header_row_idx]
                                            data = table[header_row_idx+1:]
                                        else:
                                            header = table[0]
                                            data = table[1:]
                                        
                                        # Convert to DataFrame
                                        df = pd.DataFrame(data, columns=header)
                                        
                                        # Clean up the DataFrame
                                        df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
                                        df = df.replace(r'^\s*$', pd.NA, regex=True)
                                        df = df.dropna(how='all')
                                        df = df.dropna(axis=1, how='all')
                                        
                                        if not df.empty and len(df.columns) >= 2:
                                            page_tables.append(df)
                        except Exception as e:
                            logger.debug(f"Error with extraction settings: {str(e)}")
                            continue
                    
                    # If multiple tables found on page, try to identify the invoice table
                    if len(page_tables) > 1:
                        invoice_table = self._identify_invoice_table(page_tables)
                        if invoice_table is not None:
                            extracted_tables.append(invoice_table)
                    elif len(page_tables) == 1:
                        extracted_tables.append(page_tables[0])
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {str(e)}")
        
        return extracted_tables

    def extract_with_ocr(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using OCR (Tesseract) with image preprocessing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing the extracted tables
        """
        logger.info("Extracting with OCR")
        extracted_tables = []
        
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images")
            images = convert_from_path(
                pdf_path, 
                dpi=300
            )
            
            for i, img in enumerate(images):
                logger.debug(f"Processing page {i+1} with OCR")
                
                # Convert PIL Image to OpenCV format
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
                
                # Preprocess the image
                preprocessed = preprocess_image(open_cv_image)
                
                # Detect table regions in the image
                table_regions = self.table_detector.detect_tables(open_cv_image)
                
                if not table_regions:
                    # If no tables detected, try to OCR the whole page
                    logger.info("No table regions detected, OCRing whole page")
                    try:
                        ocr_text = pytesseract.image_to_string(img)
                        logger.debug(f"OCR Text: {ocr_text[:200]}...")  # Log first 200 chars
                        
                        # Try to convert to DataFrame using pandas
                        try:
                            # Save OCR result to a temp file
                            temp_text_file = os.path.join(os.path.dirname(pdf_path), f"temp_ocr_{i}.txt")
                            with open(temp_text_file, 'w') as f:
                                f.write(ocr_text)
                            
                            # Try to parse it as structured data
                            ocr_lines = ocr_text.split('\n')
                            table_data = []
                            
                            # Look for potential header lines
                            header_idx = -1
                            for idx, line in enumerate(ocr_lines):
                                if any(keyword in line.lower() for keyword in self.header_keywords):
                                    header_idx = idx
                                    break
                            
                            if header_idx >= 0:
                                # Found a potential header
                                header = self._split_header_line(ocr_lines[header_idx])
                                
                                # Extract data rows
                                for line_idx in range(header_idx + 1, len(ocr_lines)):
                                    line = ocr_lines[line_idx].strip()
                                    if not line:
                                        continue
                                    
                                    # Split the line based on header positions
                                    row_data = self._split_line_by_positions(line, ocr_lines[header_idx], header)
                                    if row_data:
                                        table_data.append(row_data)
                                
                                if table_data:
                                    # Create DataFrame
                                    df = pd.DataFrame(table_data, columns=header)
                                    if not df.empty:
                                        extracted_tables.append(df)
                        except Exception as e:
                            logger.error(f"Error parsing OCR text: {str(e)}")
                            
                        # Try pytesseract's image_to_data approach
                        ocr_df = pytesseract.image_to_data(
                            img, output_type='data.frame',
                            config='--psm 6 -c preserve_interword_spaces=1'
                        )
                        structured_table = self._structure_ocr_data(ocr_df)
                        if structured_table is not None and not structured_table.empty:
                            extracted_tables.append(structured_table)
                    except Exception as e:
                        logger.error(f"Error during OCR processing: {str(e)}")
                else:
                    # Process each detected table region
                    logger.info(f"Found {len(table_regions)} table regions")
                    for j, (x, y, w, h) in enumerate(table_regions):
                        # Extract the table region
                        table_img = open_cv_image[y:y+h, x:x+w]
                        pil_table_img = Image.fromarray(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))
                        
                        try:
                            # OCR the table region
                            logger.debug(f"OCRing table region {j+1}")
                            ocr_df = pytesseract.image_to_data(
                                pil_table_img, output_type='data.frame',
                                config='--psm 6 -c preserve_interword_spaces=1'
                            )
                            
                            structured_table = self._structure_ocr_data(ocr_df)
                            if structured_table is not None and not structured_table.empty:
                                extracted_tables.append(structured_table)
                        except Exception as e:
                            logger.error(f"Error OCRing table region {j+1}: {str(e)}")
        
        except Exception as e:
            logger.error(f"OCR extraction error: {str(e)}")
            
        return extracted_tables

    def extract_with_hybrid_approach(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using a hybrid approach that combines pdfplumber text extraction 
        with computer vision for table structure detection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing the extracted tables
        """
        logger.info("Extracting with hybrid approach")
        extracted_tables = []
        
        try:
            # First, get all text elements with their positions from pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_num + 1} with hybrid approach")
                    
                    # Get page dimensions
                    page_width = page.width
                    page_height = page.height
                    
                    # Extract words with positions
                    words = page.extract_words(
                        keep_blank_chars=True,
                        x_tolerance=3,
                        y_tolerance=3,
                        extra_attrs=['size', 'fontname']
                    )
                    
                    if not words:
                        continue
                    
                    # Convert words to columns based on x position
                    words_df = pd.DataFrame(words)
                    
                    # Get page as image for structure detection
                    images = convert_from_path(
                        pdf_path, 
                        dpi=300, 
                        first_page=page_num+1, 
                        last_page=page_num+1
                    )
                    
                    if not images:
                        continue
                        
                    img = np.array(images[0])
                    img = img[:, :, ::-1].copy()  # Convert RGB to BGR
                    
                    # Detect table structure (rows and columns)
                    rows, cols = self.table_detector.detect_table_structure(img)
                    
                    if not rows or not cols:
                        continue
                    
                    # Map words to cells based on their position
                    table_data = self._map_words_to_cells(words_df, rows, cols, page_width, page_height)
                    
                    if table_data:
                        # Convert to dataframe
                        df = pd.DataFrame(table_data)
                        if not df.empty:
                            extracted_tables.append(df)
                    
        except Exception as e:
            logger.error(f"Hybrid extraction error: {str(e)}")
            
        return extracted_tables

    def extract_from_text(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables by analyzing raw text when other methods fail.
        Uses pattern matching and position analysis to identify table structures.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing the extracted tables
        """
        logger.info("Extracting from text")
        extracted_tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text_by_page = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        all_text_by_page.append(text)
            
            if not all_text_by_page:
                return []
            
            # Process text from each page
            for page_num, page_text in enumerate(all_text_by_page):
                logger.debug(f"Processing page {page_num + 1} text")
                
                # Split text into lines
                lines = page_text.split('\n')
                
                # Find potential header lines
                header_line_indices = []
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    keyword_count = sum(1 for keyword in self.header_keywords if keyword in line_lower)
                    if keyword_count >= 2:
                        header_line_indices.append((i, keyword_count))
                
                # Sort by keyword count (highest first)
                header_line_indices.sort(key=lambda x: x[1], reverse=True)
                
                for header_idx, _ in header_line_indices:
                    header_line = lines[header_idx]
                    
                    # Try to split the header line into columns
                    header_cols = self._split_header_line(header_line)
                    
                    if len(header_cols) >= 2:
                        # Extract data rows
                        data_rows = []
                        
                        # Process lines after the header
                        for i in range(header_idx + 1, len(lines)):
                            line = lines[i]
                            
                            # Skip empty lines
                            if not line.strip():
                                continue
                            
                            # Check if this line might be part of a table
                            # Look for numbers, especially currency amounts
                            if re.search(r'\d+\.\d{2}', line) or re.search(r'£\d+', line) or re.search(r'\$\d+', line):
                                # Try to split the line using the same strategy as the header
                                row_cols = self._split_line_by_positions(line, header_line, header_cols)
                                
                                if len(row_cols) >= 2:
                                    data_rows.append(row_cols)
                            
                            # Stop if we hit lines that look like totals or summaries
                            if re.search(r'\b(total|subtotal|sum|amount)\b', line.lower()):
                                # Check if this is actually a data row or a summary
                                row_cols = self._split_line_by_positions(line, header_line, header_cols)
                                if len(row_cols) >= 2:
                                    data_rows.append(row_cols)
                                
                                # If there are no more data rows after a few lines, we've likely hit the end of the table
                                look_ahead = min(i + 3, len(lines))
                                if not any(re.search(r'\d+\.\d{2}', lines[j]) for j in range(i + 1, look_ahead)):
                                    break
                        
                        # Create DataFrame if we have data
                        if data_rows:
                            # Determine max columns
                            max_cols = max(len(row) for row in data_rows)
                            
                            # Normalize headers and data rows to same length
                            if len(header_cols) < max_cols:
                                header_cols = header_cols + [f"Column{i+1}" for i in range(len(header_cols), max_cols)]
                            elif len(header_cols) > max_cols:
                                header_cols = header_cols[:max_cols]
                            
                            # Normalize data rows
                            normalized_rows = []
                            for row in data_rows:
                                if len(row) < max_cols:
                                    row = row + [''] * (max_cols - len(row))
                                elif len(row) > max_cols:
                                    row = row[:max_cols]
                                normalized_rows.append(row)
                            
                            # Create DataFrame
                            df = pd.DataFrame(normalized_rows, columns=header_cols)
                            
                            # Clean and normalize the DataFrame
                            df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
                            df = df.replace(r'^\s*$', pd.NA, regex=True)
                            df = df.dropna(how='all')
                            df = df.dropna(axis=1, how='all')
                            
                            if not df.empty and len(df.columns) >= 2:
                                extracted_tables.append(df)
                                
                                # If we found a good table, no need to keep processing this header
                                break
        
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
        
        return extracted_tables

    def _split_header_line(self, header_line: str) -> List[str]:
        """
        Split a header line into column names using multiple approaches.
        
        Args:
            header_line: The header line text
            
        Returns:
            List of column header names
        """
        # Try different splitting approaches
        
        # Approach 1: Split by multiple spaces
        cols = re.split(r'\s{2,}', header_line.strip())
        cols = [col.strip() for col in cols if col.strip()]
        
        if len(cols) >= 2:
            return cols
        
        # Approach 2: Look for common column headers and their positions
        header_positions = []
        for keyword in self.header_keywords:
            pos = header_line.lower().find(keyword)
            if pos >= 0:
                header_positions.append((pos, keyword))
        
        if header_positions:
            # Sort by position
            header_positions.sort()
            return [pos[1] for pos in header_positions]
        
        # Approach 3: Split by camel case and capitalization
        cols = re.findall(r'[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*', header_line)
        if len(cols) >= 2:
            return cols
            
        # Fallback: Just return the original header
        return [header_line]

    def _split_line_by_positions(self, line: str, header_line: str, header_cols: List[str]) -> List[str]:
        """
        Split a data line using the positions from the header line.
        
        Args:
            line: The data line to split
            header_line: The header line for reference
            header_cols: The split header columns
            
        Returns:
            List of column values
        """
        # Try splitting by the same character positions as the header
        row_cols = []
        
        # Build position map from header line
        positions = []
        current_pos = 0
        for col in header_cols:
            pos = header_line.find(col, current_pos)
            if pos >= 0:
                positions.append(pos)
                current_pos = pos + len(col)
        
        # Add end position
        positions.append(len(header_line))
        
        # Use positions to split the data line
        if len(positions) >= 2:
            for i in range(len(positions) - 1):
                start_pos = positions[i]
                end_pos = positions[i + 1]
                
                if start_pos < len(line):
                    if end_pos < len(line):
                        col_text = line[start_pos:end_pos].strip()
                    else:
                        col_text = line[start_pos:].strip()
                    row_cols.append(col_text)
            
            return row_cols
        
        # Fallback: Try to split by multiple spaces
        return re.split(r'\s{2,}', line.strip())

    def _structure_ocr_data(self, ocr_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process OCR data and structure it into a table.
        
        Args:
            ocr_df: DataFrame with OCR results
            
        Returns:
            Structured DataFrame or None if no table found
        """
        if ocr_df.empty:
            return None
        
        # Filter low-confidence text and empty lines
        ocr_df = ocr_df[ocr_df['conf'] > self.min_confidence]
        ocr_df = ocr_df[ocr_df['text'].notna()]
        ocr_df = ocr_df[ocr_df['text'].str.strip() != '']
        
        if ocr_df.empty:
            return None
        
        # Group words by line (using block_num and line_num)
        grouped = ocr_df.groupby(['block_num', 'line_num'])
        
        lines = []
        for _, group in grouped:
            # Sort words by their position on the line
            sorted_group = group.sort_values('left')
            line_text = ' '.join(sorted_group['text'])
            if line_text.strip():
                lines.append(line_text)
        
        if not lines:
            return None
        
        # Find potential header line
        header_line_idx = -1
        max_keyword_count = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            keyword_count = sum(1 for keyword in self.header_keywords if keyword in line_lower)
            if keyword_count > max_keyword_count:
                max_keyword_count = keyword_count
                header_line_idx = i
        
        if header_line_idx < 0 or max_keyword_count < 2:
            # Try to find lines with numbers/amounts as data rows
            data_lines = []
            for line in lines:
                if re.search(r'\d+\.\d{2}', line) or re.search(r'£\d+', line) or re.search(r'\$\d+', line):
                    data_lines.append(line)
            
            if not data_lines:
                return None
            
            # Try to create column structure based on consistent spacing
            # For now, just split by multiple spaces
            rows = [re.split(r'\s{2,}', line.strip()) for line in data_lines]
            if rows:
                max_cols = max(len(row) for row in rows)
                headers = [f"Column{i+1}" for i in range(max_cols)]
                
                # Normalize row lengths
                normalized_rows = []
                for row in rows:
                    if len(row) < max_cols:
                        row = row + [''] * (max_cols - len(row))
                    elif len(row) > max_cols:
                        row = row[:max_cols]
                    normalized_rows.append(row)
                
                return pd.DataFrame(normalized_rows, columns=headers)
        else:
            # We found a header line
            header = self._split_header_line(lines[header_line_idx])
            
            # Extract data rows
            data_rows = []
            for i in range(header_line_idx + 1, len(lines)):
                line = lines[i]
                
                # Check if this looks like a data row
                if re.search(r'\d+\.\d{2}', line) or re.search(r'£\d+', line) or re.search(r'\$\d+', line):
                    row_data = self._split_line_by_positions(line, lines[header_line_idx], header)
                    if len(row_data) >= 2:
                        data_rows.append(row_data)
            
            if data_rows:
                # Normalize data
                max_cols = max(len(row) for row in data_rows)
                if len(header) < max_cols:
                    header = header + [f"Column{i+1}" for i in range(len(header), max_cols)]
                
                # Normalize row lengths
                normalized_rows = []
                for row in data_rows:
                    if len(row) < max_cols:
                        row = row + [''] * (max_cols - len(row))
                    elif len(row) > max_cols:
                        row = row[:max_cols]
                    normalized_rows.append(row)
                
                return pd.DataFrame(normalized_rows, columns=header[:max_cols])
        
        return None

    def _map_words_to_cells(self, words_df: pd.DataFrame, rows: List[float], 
                           cols: List[float], page_width: float, page_height: float) -> List[List[str]]:
        """
        Map word positions to cells in the detected table structure.
        
        Args:
            words_df: DataFrame with word positions
            rows: Y-coordinates of row lines
            cols: X-coordinates of column lines
            page_width: Width of the page
            page_height: Height of the page
            
        Returns:
            2D list representing the table cells with text
        """
        if words_df.empty or not rows or not cols:
            return []
        
        # Normalize coordinates to the page dimensions
        rows = [r * page_height for r in rows]
        cols = [c * page_width for c in cols]
        
        # Create empty table structure
        table = [[''] for _ in range(len(rows) - 1)]
        for i in range(len(rows) - 1):
            table[i] = [''] * (len(cols) - 1)
        
        # Map each word to its cell
        for _, word in words_df.iterrows():
            x_mid = (word['x0'] + word['x1']) / 2
            y_mid = (word['top'] + word['bottom']) / 2
            
            # Find the cell this word belongs to
            row_idx = -1
            col_idx = -1
            
            for i in range(len(rows) - 1):
                if rows[i] <= y_mid < rows[i + 1]:
                    row_idx = i
                    break
            
            for j in range(len(cols) - 1):
                if cols[j] <= x_mid < cols[j + 1]:
                    col_idx = j
                    break
            
            if row_idx >= 0 and col_idx >= 0:
                # Add word to the cell
                cell_text = table[row_idx][col_idx]
                if cell_text:
                    table[row_idx][col_idx] = cell_text + ' ' + word['text']
                else:
                    table[row_idx][col_idx] = word['text']
        
        return table

    def _identify_invoice_table(self, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Identify which table is likely to be the main invoice table.
        
        Args:
            tables: List of candidate tables
            
        Returns:
            The most likely invoice table or None
        """
        if not tables:
            return None
        
        # Score each table
        table_scores = []
        
        for table in tables:
            score = self._evaluate_table_quality(table)
            table_scores.append((table, score))
        
        # Sort by score (highest first)
        table_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring table
        return table_scores[0][0] if table_scores else None

    def _evaluate_table_quality(self, table: pd.DataFrame) -> float:
        """
        Evaluate the quality of an extracted table.
        
        Args:
            table: DataFrame to evaluate
            
        Returns:
            Quality score (higher is better)
        """
        if table is None or table.empty:
            return 0.0
        
        score = 0.0
        
        # 1. Number of columns (invoice tables usually have 3+ columns)
        num_cols = len(table.columns)
        if num_cols >= 4:
            score += 3.0
        elif num_cols >= 3:
            score += 2.0
        elif num_cols >= 2:
            score += 1.0
        
        # 2. Number of rows (more rows usually means more line items)
        num_rows = len(table)
        if num_rows >= 5:
            score += 3.0
        elif num_rows >= 3:
            score += 2.0
        elif num_rows >= 1:
            score += 1.0
        
        # 3. Presence of key column headers
        headers = [str(col).lower() for col in table.columns]
        header_score = sum(1 for keyword in self.header_keywords if any(keyword in header for header in headers))
        score += min(header_score, 5.0)  # Cap at 5 points
        
        # 4. Presence of numerical data (prices, quantities)
        # Count cells with numerical values
        num_cells = 0
        for col in table.columns:
            for val in table[col]:
                if isinstance(val, str) and re.search(r'\d+\.\d{2}', val):
                    num_cells += 1
                elif isinstance(val, (int, float)):
                    num_cells += 1
        
        if num_cells >= 10:
            score += 4.0
        elif num_cells >= 5:
            score += 2.0
        elif num_cells >= 1:
            score += 1.0
        
        # 5. Presence of currency symbols
        currency_count = 0
        for col in table.columns:
            for val in table[col]:
                if isinstance(val, str) and re.search(r'[$£€]', val):
                    currency_count += 1
        
        if currency_count >= 5:
            score += 3.0
        elif currency_count >= 1:
            score += 1.0
        
        # 6. Consistency of data types within columns
        consistency_score = 0
        for col in table.columns:
            # Try to infer data type
            numeric_count = sum(1 for val in table[col] if isinstance(val, (int, float)) or 
                              (isinstance(val, str) and re.search(r'\d+\.\d{2}', val)))
            text_count = sum(1 for val in table[col] if isinstance(val, str) and not re.search(r'\d+\.\d{2}', val))
            
            # If column is mostly one type, increase score
            if numeric_count > 0.7 * len(table) or text_count > 0.7 * len(table):
                consistency_score += 1
        
        score += min(consistency_score, 3.0)  # Cap at 3 points
        
        return score

    def _post_process_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process the extracted table to clean and normalize data.
        
        Args:
            table: DataFrame to process
            
        Returns:
            Cleaned and normalized DataFrame
        """
        if table is None or table.empty:
            return pd.DataFrame()
        
        # 1. Clean column names
        table.columns = [clean_text(str(col)) for col in table.columns]
        
        # 2. Remove duplicate columns
        table = table.loc[:, ~table.columns.duplicated()]
        
        # 3. Handle merged cells (rows with NaN in some columns)
        table = postprocess_dataframe(table)
        
        # 4. Remove summary rows (often at the bottom, containing totals)
        # Identify rows containing summary keywords
        summary_rows = []
        for i, row in table.iterrows():
            row_str = ' '.join(str(val) for val in row.values)
            if re.search(r'\b(total|subtotal|sum|amount)\b', row_str.lower()):
                # Check if this row has different patterns than data rows
                # For example, fewer columns with values, or positioned at the end
                if i > len(table) * 0.7:  # If in the last 30% of rows
                    summary_rows.append(i)
        
        # Drop summary rows if they're at the end
        if summary_rows and all(i >= len(table) * 0.7 for i in summary_rows):
            table = table.drop(summary_rows)
        
        # 5. Remove columns that are mostly empty
        threshold = 0.8  # If 80% or more values are empty, drop the column
        min_count = int((1.0 - threshold) * len(table))
        table = table.dropna(axis=1, thresh=min_count)
        
        # 6. Handle multi-line text in cells
        for col in table.columns:
            table[col] = table[col].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
        
        # 7. Trim whitespace
        for col in table.columns:
            table[col] = table[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # 8. Try to detect and fix misaligned columns
        # If a column has values that look like they belong to another column, fix it
        for i, col in enumerate(table.columns[:-1]):
            next_col = table.columns[i+1]
            
            # Check if current column values have unit patterns (e.g., price/unit)
            # and next column has quantity patterns
            current_col_has_prices = sum(1 for val in table[col] if isinstance(val, str) and 
                                        re.search(r'\$|\£|\€|\d+\.\d{2}', val))
            next_col_has_quantities = sum(1 for val in table[next_col] if isinstance(val, (int, float)) or
                                        (isinstance(val, str) and re.search(r'\b\d+\b', val)))
            
            # If they seem misaligned, shift the values
            if current_col_has_prices > 0.5 * len(table) and next_col_has_quantities > 0.5 * len(table):
                # Store current values
                current_values = table[col].copy()
                next_values = table[next_col].copy()
                
                # Shift values right by one column for rows where it makes sense
                for j in range(len(table)):
                    current_val = str(table.loc[j, col]) if not pd.isna(table.loc[j, col]) else ""
                    next_val = str(table.loc[j, next_col]) if not pd.isna(table.loc[j, next_col]) else ""
                    
                    # If current has price pattern and next has quantity pattern, swap them
                    if (re.search(r'\$|\£|\€|\d+\.\d{2}', current_val) and 
                        re.search(r'\b\d+\b', next_val) and not re.search(r'\d+\.\d{2}', next_val)):
                        table.loc[j, col] = next_values[j]
                        table.loc[j, next_col] = current_values[j]
        
        return table

    def _save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"CSV saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
