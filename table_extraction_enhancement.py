"""
Enhanced table extraction module for complex PDF invoices.
This module provides additional methods to improve extraction accuracy.
"""

import re
import os
import logging
import numpy as np
import pandas as pd
import cv2
from pdf2image import convert_from_path
import pdfplumber
import pytesseract
from typing import List, Dict, Tuple, Optional, Union, Any, Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTableExtractor:
    """
    Advanced techniques for extracting tables from complex PDF invoices.
    """
    
    def __init__(self, debug=False):
        """Initialize the enhanced table extractor."""
        self.debug = debug
        
        # Common patterns that indicate table headers in invoices
        self.header_patterns = {
            'item': r'(?i)(?:^|\s)(item|description|product|article|service|particulars)(?:$|\s|s$)',
            'quantity': r'(?i)(?:^|\s)(qty|quantity|items|pcs|units|count)(?:$|\s)',
            'unit': r'(?i)(?:^|\s)(unit|uom|measure|pkg)(?:$|\s)',
            'price': r'(?i)(?:^|\s)(price|rate|unit price|cost|unit cost|price/rate)(?:$|\s|/)',
            'amount': r'(?i)(?:^|\s)(amount|total|line amount|ext|extension|value|line value|net|sub\s*total)(?:$|\s)',
            'vat': r'(?i)(?:^|\s)(vat|tax|gst|hst|duty|vat\s*%|vat\s*rate)(?:$|\s|%|\()',
            'sku': r'(?i)(?:^|\s)(sku|code|ref|item\s*no|art|reference|product\s*code)(?:$|\s|\.)',
        }
        
        # Regular expressions for monetary values
        self.money_pattern = r'(?:\£|\$|€|₹|USD|GBP|EUR|Rs|Rs\.)*\s*(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\.\d{2})?'
        
        # Patterns that indicate the end of a table
        self.table_end_patterns = [
            r'(?i)total',
            r'(?i)sub\s*total',
            r'(?i)invoice\s*total',
            r'(?i)amount\s*due',
            r'(?i)balance\s*due',
        ]
        
        # Common column names and their variations for standardization
        self.column_name_mapping = {
            # Description variations
            'description': ['desc', 'item', 'product', 'article', 'service', 'particulars', 'item name', 'details'],
            
            # Quantity variations
            'quantity': ['qty', 'units', 'pcs', 'pieces', 'count', 'items', 'ordered', 'delivered'],
            
            # Unit variations
            'unit': ['uom', 'unit of measure', 'measure', 'pkg'],
            
            # Price variations
            'unit_price': ['price', 'rate', 'unit price', 'cost', 'unit cost', 'price/rate', 'unit rate', 'price per unit'],
            
            # VAT variations
            'vat_rate': ['vat', 'vat %', 'tax', 'tax rate', 'gst', 'hst', 'duty', 'tax %', 'vat rate', 'gst rate'],
            
            # Amount variations
            'amount': ['total', 'line amount', 'ext', 'extension', 'value', 'line value', 'net', 'line total', 'extended'],
            
            # Code variations
            'code': ['sku', 'ref', 'item no', 'art', 'reference', 'product code', 'item code', 'no.']
        }

    def extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract all potential tables from a PDF using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables as pandas DataFrames
        """
        logger.info(f"Extracting tables from: {pdf_path}")
        
        tables = []
        
        # Method 1: Try pdfplumber with various settings
        logger.info("Trying pdfplumber extraction...")
        plumber_tables = self._extract_with_pdfplumber(pdf_path)
        if plumber_tables:
            tables.extend(plumber_tables)
        
        # Method 2: Try layout analysis
        logger.info("Trying layout analysis extraction...")
        layout_tables = self._extract_with_layout_analysis(pdf_path)
        if layout_tables:
            tables.extend(layout_tables)
            
        # Method 3: Try OCR if other methods didn't yield good results
        if not tables or max(len(df) for df in tables if df is not None and not df.empty) < 5:
            logger.info("Trying OCR extraction...")
            ocr_tables = self._extract_with_ocr(pdf_path)
            if ocr_tables:
                tables.extend(ocr_tables)
        
        # Method 4: Try text pattern analysis as a last resort
        if not tables or max(len(df) for df in tables if df is not None and not df.empty) < 3:
            logger.info("Trying text pattern analysis...")
            pattern_tables = self._extract_with_text_patterns(pdf_path)
            if pattern_tables:
                tables.extend(pattern_tables)
        
        # Filter out None and empty tables
        tables = [df for df in tables if df is not None and not df.empty]
        
        # Post-process all tables
        processed_tables = []
        for i, table in enumerate(tables):
            try:
                processed = self._post_process_table(table)
                if not processed.empty and len(processed.columns) >= 2:
                    processed_tables.append(processed)
            except Exception as e:
                logger.warning(f"Error post-processing table {i}: {str(e)}")
        
        logger.info(f"Extracted {len(processed_tables)} potential tables")
        return processed_tables
    
    def select_best_invoice_table(self, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Select the best table from the extracted candidates based on invoice characteristics.
        
        Args:
            tables: List of candidate tables
            
        Returns:
            The most likely invoice table or None if no good candidates
        """
        if not tables:
            return None
        
        # Score each table based on how likely it is to be the main invoice table
        scored_tables = []
        
        for table in tables:
            score = 0
            
            # 1. Number of rows (more is better, up to a point)
            num_rows = len(table)
            if num_rows >= 10:
                score += 5
            elif num_rows >= 5:
                score += 3
            elif num_rows >= 3:
                score += 1
            
            # 2. Number of columns (invoice tables typically have 3-7 columns)
            num_cols = len(table.columns)
            if 4 <= num_cols <= 7:
                score += 5
            elif 3 <= num_cols <= 8:
                score += 3
            elif num_cols > 8:
                score += 1
            
            # 3. Presence of key invoice column headers
            columns_lower = [str(col).lower() for col in table.columns]
            col_text = ' '.join(columns_lower)
            
            for pattern_type, pattern in self.header_patterns.items():
                if any(re.search(pattern, col) for col in columns_lower) or re.search(pattern, col_text):
                    score += 2
            
            # 4. Contains numeric/monetary values (essential for invoices)
            num_numeric_cells = 0
            for col in table.columns:
                # Check for monetary values in column header
                if re.search(self.money_pattern, str(col)):
                    score += 1
                
                # Check cells for monetary values
                num_money_cells = 0
                for val in table[col]:
                    if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(self.money_pattern, val)):
                        num_money_cells += 1
                        num_numeric_cells += 1
                
                # If more than half of the cells contain monetary values, likely a price/amount column
                if num_money_cells > len(table) / 2:
                    score += 3
            
            if num_numeric_cells >= num_rows:  # At least one numeric value per row
                score += 3
            
            # 5. Has consistent formatting across rows (good sign of a table)
            if self._check_row_consistency(table):
                score += 3
            
            # 6. Penalize tables with too many NaN values
            nan_ratio = table.isna().sum().sum() / (len(table) * len(table.columns))
            if nan_ratio > 0.3:
                score -= int(nan_ratio * 10)
            
            scored_tables.append((table, score))
            
            if self.debug:
                logger.debug(f"Table with {num_rows} rows, {num_cols} cols received score: {score}")
        
        # Sort by score (highest first)
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug and scored_tables:
            logger.debug(f"Best table score: {scored_tables[0][1]}")
        
        # Return highest scoring table, or None if all scores are very low
        if scored_tables and scored_tables[0][1] > 5:
            return scored_tables[0][0]
        
        # If no table has a good score but we have candidates, return the one with most rows
        if tables:
            return max(tables, key=len)
            
        return None
    
    def _check_row_consistency(self, df: pd.DataFrame) -> bool:
        """Check if table has consistent formatting across rows (good sign of a true table)."""
        if len(df) < 3:
            return False
        
        # Check if numeric columns have consistent numeric values
        consistent_cols = 0
        for col in df.columns:
            numeric_count = 0
            for val in df[col]:
                if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r'\d+(\.\d+)?', val)):
                    numeric_count += 1
            
            # If more than 75% of values are numeric, consider it consistent
            if numeric_count > len(df) * 0.75 or numeric_count == 0:
                consistent_cols += 1
        
        # If more than half of columns are consistent, consider the table consistent
        return consistent_cols > len(df.columns) / 2
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber with multiple settings."""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    if self.debug:
                        logger.debug(f"Processing page {page_idx + 1} with pdfplumber")
                    
                    # Try multiple table settings
                    table_settings_list = [
                        {},  # Default
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                        {
                            "vertical_strategy": "text", 
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                            "min_words_vertical": 2,
                            "min_words_horizontal": 2
                        },
                        {
                            "vertical_strategy": "lines_strict", 
                            "horizontal_strategy": "lines_strict",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 5
                        },
                        # More aggressive text-based extraction for invisible tables
                        {
                            "vertical_strategy": "text", 
                            "horizontal_strategy": "text",
                            "snap_tolerance": 10,
                            "join_tolerance": 5,
                            "min_words_vertical": 1,
                            "min_words_horizontal": 1
                        }
                    ]
                    
                    for settings in table_settings_list:
                        try:
                            # Extract tables with current settings
                            found_tables = page.extract_tables(table_settings=settings)
                            
                            for table_data in found_tables:
                                if table_data and len(table_data) > 1:  # At least a header + one data row
                                    # Identify potential header row
                                    header_idx = self._find_header_row(table_data)
                                    
                                    if header_idx >= 0:
                                        # Use identified header
                                        header = table_data[header_idx]
                                        data = table_data[header_idx+1:]
                                    else:
                                        # Default to first row as header
                                        header = table_data[0]
                                        data = table_data[1:]
                                    
                                    # Convert to DataFrame
                                    if data:
                                        # Handle duplicate column names
                                        clean_header = self._clean_and_normalize_headers(header)
                                        
                                        df = pd.DataFrame(data, columns=clean_header)
                                        if not df.empty and len(df.columns) >= 2:
                                            tables.append(df)
                        
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error with table settings {settings}: {str(e)}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in pdfplumber extraction: {str(e)}")
        
        return tables
    
    def _extract_with_layout_analysis(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables by analyzing text layout and positioning.
        This is effective for tables without explicit borders.
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    if self.debug:
                        logger.debug(f"Processing page {page_idx + 1} with layout analysis")
                    
                    # Get all text elements with their positions
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    
                    if not words:
                        continue
                    
                    # Group words by y-position (lines)
                    line_groups = {}
                    for word in words:
                        # Use rounded y-position as key for grouping
                        y_key = round(word['top'] / 2) * 2  # Group within 2 points
                        if y_key not in line_groups:
                            line_groups[y_key] = []
                        line_groups[y_key].append(word)
                    
                    # Sort lines by y-position
                    sorted_lines = [line_groups[y] for y in sorted(line_groups.keys())]
                    
                    # Look for potential header lines
                    header_candidates = []
                    for line_idx, line in enumerate(sorted_lines):
                        line_text = ' '.join(word['text'] for word in line)
                        
                        # Check if this line contains multiple header patterns
                        pattern_count = sum(1 for pattern in self.header_patterns.values() 
                                           if re.search(pattern, line_text.lower()))
                        
                        if pattern_count >= 2:
                            header_candidates.append((line_idx, pattern_count, line))
                    
                    # Sort by pattern count (highest first)
                    header_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Process each header candidate
                    for header_idx, _, header_line in header_candidates:
                        try:
                            # Define a table starting from this header
                            table_lines = []
                            
                            # Sort header words by x-position to establish column positions
                            sorted_header_words = sorted(header_line, key=lambda w: w['x0'])
                            header_positions = [w['x0'] for w in sorted_header_words]
                            header_texts = [w['text'] for w in sorted_header_words]
                            
                            # Add header row
                            table_lines.append(header_texts)
                            
                            # Find data rows (lines after header)
                            in_table = True
                            for line_idx in range(header_idx + 1, len(sorted_lines)):
                                current_line = sorted_lines[line_idx]
                                line_text = ' '.join(word['text'] for word in current_line)
                                
                                # Check if we've reached the end of the table
                                if any(re.search(pattern, line_text) for pattern in self.table_end_patterns):
                                    # Include this line (might contain totals) and then stop
                                    data_row = self._map_line_to_columns(current_line, header_positions)
                                    table_lines.append(data_row)
                                    in_table = False
                                    break
                                
                                # Check if this line is too different from the header (might indicate end of table)
                                # For example, too few words or very different position pattern
                                if len(current_line) < len(header_texts) / 2:
                                    continue
                                
                                # Map words to columns based on x-positions
                                data_row = self._map_line_to_columns(current_line, header_positions)
                                table_lines.append(data_row)
                            
                            # If we captured data rows, convert to DataFrame
                            if len(table_lines) > 1:
                                # Clean up header
                                header = self._clean_and_normalize_headers(table_lines[0])
                                data = table_lines[1:]
                                
                                # Ensure all rows have same number of columns
                                max_cols = max(len(row) for row in data)
                                max_cols = max(max_cols, len(header))
                                
                                # Pad header if needed
                                if len(header) < max_cols:
                                    header.extend([f'Column{i+1}' for i in range(len(header), max_cols)])
                                
                                # Pad data rows
                                for i in range(len(data)):
                                    if len(data[i]) < max_cols:
                                        data[i].extend([''] * (max_cols - len(data[i])))
                                
                                df = pd.DataFrame(data, columns=header[:max_cols])
                                if not df.empty and len(df.columns) >= 2:
                                    tables.append(df)
                        
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error processing header candidate: {str(e)}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in layout analysis: {str(e)}")
        
        return tables
    
    def _map_line_to_columns(self, line_words, header_positions):
        """Map words in a line to columns based on header positions."""
        # Sort words by x-position
        sorted_words = sorted(line_words, key=lambda w: w['x0'])
        
        # Initialize result with empty strings
        result = [''] * len(header_positions)
        
        # Map each word to the closest header position
        for word in sorted_words:
            # Find closest header position
            distances = [abs(word['x0'] - pos) for pos in header_positions]
            closest_idx = distances.index(min(distances))
            
            # Append to that column (in case multiple words map to same column)
            if result[closest_idx]:
                result[closest_idx] += ' ' + word['text']
            else:
                result[closest_idx] = word['text']
        
        return result
    
    def _extract_with_ocr(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using OCR for scanned or image-based PDFs."""
        tables = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)
            
            for page_idx, img in enumerate(images):
                if self.debug:
                    logger.debug(f"Processing page {page_idx + 1} with OCR")
                
                # Convert PIL image to OpenCV format
                img_cv = np.array(img)
                img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
                
                # Preprocess image
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
                
                # Detect table structure using horizontal and vertical lines
                horizontal = self._detect_horizontal_lines(thresh)
                vertical = self._detect_vertical_lines(thresh)
                
                # Combine to get table grid
                table_grid = cv2.bitwise_or(horizontal, vertical)
                
                # Find contours to identify table cells
                contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # If contours were found, try to extract table structure
                if contours and len(contours) > 10:  # Need enough cells for a table
                    # Extract text from full image
                    ocr_text = pytesseract.image_to_string(img)
                    lines = ocr_text.split('\n')
                    
                    # Use contours to determine potential row/column structure
                    bboxes = [cv2.boundingRect(c) for c in contours]
                    
                    # Sort by y-coordinate to group by rows
                    row_sorted = sorted(bboxes, key=lambda x: x[1])
                    
                    # Group by rows (y-coordinate)
                    row_groups = []
                    current_row = [row_sorted[0]]
                    for bbox in row_sorted[1:]:
                        if abs(bbox[1] - current_row[0][1]) < 20:  # Threshold for same row
                            current_row.append(bbox)
                        else:
                            # Sort cells in row by x-coordinate
                            row_groups.append(sorted(current_row, key=lambda x: x[0]))
                            current_row = [bbox]
                    
                    if current_row:
                        row_groups.append(sorted(current_row, key=lambda x: x[0]))
                    
                    # Now extract text for each cell using OCR
                    table_data = []
                    for row in row_groups:
                        row_data = []
                        for x, y, w, h in row:
                            # Extract cell image
                            cell_img = img_cv[y:y+h, x:x+w]
                            # OCR on cell
                            cell_text = pytesseract.image_to_string(cell_img).strip()
                            row_data.append(cell_text)
                        
                        if row_data:
                            table_data.append(row_data)
                
                # If no table structure was detected, try to infer from text patterns
                if not contours or len(contours) <= 10:
                    # Extract text from full image
                    ocr_text = pytesseract.image_to_string(img)
                    lines = ocr_text.split('\n')
                    
                    # Remove empty lines
                    lines = [line.strip() for line in lines if line.strip()]
                    
                    # Look for potential header lines
                    for i, line in enumerate(lines):
                        line_lower = line.lower()
                        # Check if this line contains multiple header patterns
                        pattern_count = sum(1 for pattern in self.header_patterns.values() 
                                           if re.search(pattern, line_lower))
                        
                        if pattern_count >= 2:
                            # Found potential header, extract table
                            table_data = []
                            
                            # Identify columns by spaces or delimiters
                            if '|' in line:
                                # Table uses | as delimiter
                                delimiter = '|'
                            else:
                                # Try to split by runs of whitespace
                                delimiter = None  # Will use regex later
                            
                            # Extract header
                            if delimiter:
                                header = [col.strip() for col in line.split(delimiter) if col.strip()]
                            else:
                                # Split by multiple spaces
                                header = re.split(r'\s{2,}', line)
                                header = [col.strip() for col in header if col.strip()]
                            
                            table_data.append(header)
                            
                            # Extract data rows
                            for j in range(i+1, len(lines)):
                                data_line = lines[j]
                                
                                # Check if we've reached the end of the table
                                if any(re.search(pattern, data_line.lower()) for pattern in self.table_end_patterns):
                                    break
                                
                                # Extract row data using same delimiter as header
                                if delimiter:
                                    row = [col.strip() for col in data_line.split(delimiter) if col.strip()]
                                else:
                                    # Split by multiple spaces or try to align with header positions
                                    row = re.split(r'\s{2,}', data_line)
                                    row = [col.strip() for col in row if col.strip()]
                                
                                if row:
                                    table_data.append(row)
                            
                            # If we found enough data rows, convert to DataFrame
                            if len(table_data) > 1:
                                try:
                                    # Normalize row lengths
                                    max_cols = max(len(row) for row in table_data)
                                    for i in range(len(table_data)):
                                        if len(table_data[i]) < max_cols:
                                            table_data[i].extend([''] * (max_cols - len(table_data[i])))
                                    
                                    # Clean up header
                                    header = self._clean_and_normalize_headers(table_data[0])
                                    
                                    # Create DataFrame
                                    df = pd.DataFrame(table_data[1:], columns=header)
                                    if not df.empty and len(df.columns) >= 2:
                                        tables.append(df)
                                
                                except Exception as e:
                                    if self.debug:
                                        logger.debug(f"Error converting OCR data to DataFrame: {str(e)}")
                                
                            # Only process the first good header we find
                            break
        
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
        
        return tables
    
    def _detect_horizontal_lines(self, img):
        """Detect horizontal lines in an image."""
        # Get image dimensions
        h, w = img.shape
        
        # Create horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
        
        # Detect horizontal lines
        horizontal = cv2.erode(img, horizontal_kernel, iterations=3)
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=3)
        
        return horizontal
    
    def _detect_vertical_lines(self, img):
        """Detect vertical lines in an image."""
        # Get image dimensions
        h, w = img.shape
        
        # Create vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))
        
        # Detect vertical lines
        vertical = cv2.erode(img, vertical_kernel, iterations=3)
        vertical = cv2.dilate(vertical, vertical_kernel, iterations=3)
        
        return vertical
    
    def _extract_with_text_patterns(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables based on text patterns (for challenging PDFs)."""
        tables = []
        
        try:
            # Extract all text from PDF
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n\n"
            
            # Split into lines
            lines = all_text.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            # Find potential table regions (between header lines and table end markers)
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check if this is a potential header line
                pattern_count = sum(1 for pattern in self.header_patterns.values() 
                                   if re.search(pattern, line_lower))
                
                if pattern_count >= 2:
                    # Found potential header, extract table
                    table_data = []
                    
                    # Try to identify column separator pattern
                    if '|' in line:
                        delimiter = '|'
                    else:
                        delimiter = None  # Will use regex to split by whitespace
                    
                    # Extract header
                    if delimiter:
                        header = [col.strip() for col in line.split(delimiter) if col.strip()]
                    else:
                        # Try to split by position (multiple spaces)
                        header = self._split_by_whitespace(line)
                    
                    table_data.append(header)
                    
                    # Extract data rows
                    for j in range(i+1, len(lines)):
                        data_line = lines[j]
                        
                        # Check if we've reached the end of the table
                        if any(re.search(pattern, data_line.lower()) for pattern in self.table_end_patterns):
                            # Include this line (might contain totals) then stop
                            if delimiter:
                                row = [col.strip() for col in data_line.split(delimiter) if col.strip()]
                            else:
                                row = self._split_by_whitespace(data_line)
                            
                            if row:
                                table_data.append(row)
                            break
                        
                        # Skip lines that might be footnotes or annotations
                        if len(data_line) < 10 or data_line.startswith('*') or data_line.startswith('Note'):
                            continue
                        
                        # Extract row using same method as header
                        if delimiter:
                            row = [col.strip() for col in data_line.split(delimiter) if col.strip()]
                        else:
                            row = self._split_by_whitespace(data_line)
                        
                        if row:
                            table_data.append(row)
                    
                    # If we found data rows, convert to DataFrame
                    if len(table_data) > 1:
                        try:
                            # Normalize row lengths
                            max_cols = max(len(row) for row in table_data)
                            for i in range(len(table_data)):
                                if len(table_data[i]) < max_cols:
                                    table_data[i].extend([''] * (max_cols - len(table_data[i])))
                            
                            # Clean up header
                            header = self._clean_and_normalize_headers(table_data[0])
                            
                            # Create DataFrame
                            df = pd.DataFrame(table_data[1:], columns=header)
                            if not df.empty and len(df.columns) >= 2:
                                tables.append(df)
                        
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error converting pattern data to DataFrame: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in text pattern extraction: {str(e)}")
        
        return tables
    
    def _split_by_whitespace(self, line: str) -> List[str]:
        """Split a line by runs of whitespace, intelligently handling values with spaces."""
        # First try naive split by multiple spaces
        parts = re.split(r'\s{2,}', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        # If we got a reasonable number of parts, use them
        if len(parts) >= 3:
            return parts
        
        # Otherwise try more aggressive splitting
        parts = re.split(r'\s+', line)
        return [p.strip() for p in parts if p.strip()]
    
    def _find_header_row(self, table_data: List[List[str]]) -> int:
        """Find the most likely header row in a table."""
        if not table_data:
            return -1
        
        max_score = -1
        header_idx = -1
        
        # Check each row for header-like characteristics
        for i, row in enumerate(table_data):
            if not row:
                continue
            
            # Combine row text for pattern matching
            row_text = ' '.join(str(cell).lower() for cell in row if cell)
            
            # Count header patterns
            pattern_count = sum(1 for pattern in self.header_patterns.values() 
                               if any(re.search(pattern, str(cell).lower()) for cell in row) 
                               or re.search(pattern, row_text))
            
            # Presence of capital letters (headers often have proper casing)
            capital_count = sum(1 for cell in row if isinstance(cell, str) and any(c.isupper() for c in cell))
            
            # Calculate score
            score = pattern_count * 2 + capital_count
            
            # Headers rarely contain long text or numeric values
            long_text = sum(1 for cell in row if isinstance(cell, str) and len(cell) > 30)
            numeric_values = sum(1 for cell in row if isinstance(cell, (int, float)) or 
                                (isinstance(cell, str) and re.match(r'^[\d,.]+$', cell)))
            
            score -= (long_text + numeric_values)
            
            if score > max_score:
                max_score = score
                header_idx = i
        
        # Return the index if the score is reasonable
        return header_idx if max_score >= 2 else 0  # Default to first row if no good candidate
    
    def _clean_and_normalize_headers(self, headers: List[str]) -> List[str]:
        """Clean and normalize header names, handling duplicates."""
        if not headers:
            return []
        
        # Clean each header
        clean_headers = []
        for h in headers:
            # Convert to string and clean
            header = str(h).strip()
            
            # Remove common prefixes/suffixes
            header = re.sub(r'^[-_:#*.\s]+|[-_:#*.\s]+$', '', header)
            
            # Normalize whitespace
            header = re.sub(r'\s+', ' ', header)
            
            clean_headers.append(header if header else 'Column')
        
        # Handle duplicate headers
        seen = set()
        for i in range(len(clean_headers)):
            original = clean_headers[i]
            current = original
            counter = 1
            
            # Append number until unique
            while current in seen:
                current = f"{original}_{counter}"
                counter += 1
            
            clean_headers[i] = current
            seen.add(current)
        
        return clean_headers
    
    def _post_process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize extracted table data."""
        if df.empty:
            return df
        
        # 1. Try to identify and standardize column names
        std_columns = {}
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check each type of column
            for std_name, variations in self.column_name_mapping.items():
                if any(var in col_lower for var in variations):
                    std_columns[col] = std_name
                    break
        
        # Rename columns if standards were found
        if std_columns:
            df = df.rename(columns=std_columns)
        
        # 2. Handle multi-line column names (merge cell content if it looks like continuation)
        for col in df.columns:
            prev_val = None
            merged_indices = []
            
            for i, val in enumerate(df[col]):
                if isinstance(val, str):
                    # Check if this looks like continuation (no numbers, short text)
                    if prev_val is not None and not re.search(r'\d', val) and len(val) < 20:
                        # This might be continuation of previous line
                        df.at[i-1, col] = f"{prev_val} {val}"
                        merged_indices.append(i)
                
                prev_val = val
            
            # Remove merged rows
            if merged_indices:
                df = df.drop(merged_indices)
        
        # 3. Remove rows that are likely not part of the table
        # (e.g., rows with too many empty cells or footnotes)
        rows_to_drop = []
        for i, row in df.iterrows():
            empty_count = row.isna().sum()
            if empty_count > len(df.columns) * 0.7:  # More than 70% empty
                rows_to_drop.append(i)
            elif isinstance(row.iloc[0], str) and row.iloc[0].startswith(('*', 'Note:')):
                rows_to_drop.append(i)
        
        if rows_to_drop:
            df = df.drop(rows_to_drop)
        
        # 4. Normalize numeric and currency values
        for col in df.columns:
            # Check if this column has currency/numeric values
            money_count = 0
            for val in df[col]:
                if isinstance(val, str) and re.search(self.money_pattern, val):
                    money_count += 1
            
            # If more than 30% of values are money/numeric, normalize the column
            if money_count > len(df) * 0.3:
                df[col] = df[col].apply(self._normalize_numeric)
        
        # 5. Reset index if rows were dropped
        if rows_to_drop:
            df = df.reset_index(drop=True)
        
        return df
    
    def _normalize_numeric(self, val):
        """Normalize numeric and currency values to standard format."""
        if not isinstance(val, str):
            return val
        
        # Remove currency symbols and separators
        clean_val = re.sub(r'[£$€₹,\s]', '', val)
        
        # Try to convert to numeric
        try:
            if '.' in clean_val:
                return float(clean_val)
            else:
                return int(clean_val)
        except ValueError:
            return val