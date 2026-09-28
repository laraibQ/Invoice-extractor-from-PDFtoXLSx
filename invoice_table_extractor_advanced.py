#!/usr/bin/env python3
"""
Advanced Invoice Table Extractor

This script provides specialized extraction for invoice tables from PDF files.
It's designed to handle the specific formats and layouts common in invoices.

Usage:
    python invoice_table_extractor_advanced.py invoice.pdf [output.csv]
"""

import os
import re
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import pdfplumber
import pytesseract
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedInvoiceExtractor:
    """
    Specialized extractor for invoice tables with multiple strategies
    optimized for different invoice layouts and formats.
    """
    
    def __init__(self, tesseract_path=None, debug=False):
        """
        Initialize the invoice extractor with necessary configurations.
        
        Args:
            tesseract_path: Path to tesseract executable (for OCR)
            debug: Enable debug mode with detailed logging
        """
        self.debug = debug
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Invoice-specific patterns for detecting relevant table regions
        self.table_header_patterns = [
            # Product/item column indicators
            r'(?i)\b(item|description|product|article|service|particulars)\b',
            # Quantity column indicators
            r'(?i)\b(qty|quantity|units|pcs|count)\b',
            # Price column indicators
            r'(?i)\b(price|rate|unit[\s-]?price|cost)\b',
            # Amount column indicators
            r'(?i)\b(amount|total|line[\s-]?amount|extended|value|net)\b',
            # VAT/tax column indicators
            r'(?i)\b(vat|tax|gst|hst|vat[\s-]?%|tax[\s-]?rate)\b'
        ]
        
        # End-of-table indicators (often appear after the main table)
        self.table_end_patterns = [
            r'(?i)\b(subtotal|sub[\s-]?total)\b',
            r'(?i)\b(total|invoice[\s-]?total|amount[\s-]?due)\b',
            r'(?i)\b(vat[\s-]?total|tax[\s-]?total)\b',
            r'(?i)\b(balance[\s-]?due|outstanding[\s-]?balance)\b'
        ]
        
        # Patterns for value formatting (especially monetary values)
        self.money_pattern = r'[$£€]\s*[\d,]+\.\d{2}|\d+\.\d{2}|\d{1,3}(?:,\d{3})+(?:\.\d{2})?'
        
        # Common separators in invoice tables
        self.common_separators = ['|', '\t', '  ']
        
        # Standard column names for normalization
        self.standard_columns = {
            'item': ['item', 'sku', 'code', 'ref', 'no.', 'article'],
            'description': ['description', 'desc', 'product', 'service', 'particulars', 'item name'],
            'quantity': ['quantity', 'qty', 'units', 'pcs', 'count'],
            'unit': ['unit', 'uom', 'measure'],
            'price': ['price', 'unit price', 'rate', 'cost', 'unit cost'],
            'vat': ['vat', 'tax', 'gst', 'hst', 'vat %', 'tax rate'],
            'amount': ['amount', 'total', 'line amount', 'extended', 'value', 'net', 'line total']
        }
    
    def extract_from_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to extract invoice table from PDF.
        
        Args:
            pdf_path: Path to the invoice PDF
            output_path: Path to save the CSV output (if None, doesn't save)
            
        Returns:
            DataFrame containing the extracted table
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing invoice: {pdf_path}")
        
        # Try each extraction method in order of reliability and performance
        try:
            # Method 1: PDF native text extraction with pdfplumber
            logger.info("Attempting structured extraction with pdfplumber...")
            tables = self._extract_with_structured_method(pdf_path)
            
            # Method 2: Layout-based extraction
            if not tables or not self._is_table_valid(tables[0]):
                logger.info("Structured extraction failed, trying layout-based extraction...")
                tables = self._extract_with_layout_method(pdf_path)
            
            # Method 3: OCR-based extraction for scanned/image invoices
            if not tables or not self._is_table_valid(tables[0]):
                logger.info("Layout extraction failed, trying OCR-based extraction...")
                tables = self._extract_with_ocr_method(pdf_path)
            
            # Method 4: Text pattern analysis as last resort
            if not tables or not self._is_table_valid(tables[0]):
                logger.info("OCR extraction failed, trying text pattern analysis...")
                tables = self._extract_with_text_patterns(pdf_path)
            
            # If all methods failed, return empty DataFrame
            if not tables or all(table.empty for table in tables):
                logger.warning("All extraction methods failed to find valid tables")
                return pd.DataFrame()
            
            # Select the best table based on invoice-specific criteria
            best_table = self._select_best_invoice_table(tables)
            
            if best_table is None or best_table.empty:
                logger.warning("No valid invoice table found")
                return pd.DataFrame()
            
            # Post-process the table to clean and normalize it
            processed_table = self._post_process_table(best_table)
            
            # Save to CSV if output path provided
            if output_path:
                processed_table.to_csv(output_path, index=False)
                logger.info(f"Table saved to: {output_path}")
            
            return processed_table
        
        except Exception as e:
            logger.error(f"Error extracting table: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return pd.DataFrame()
    
    def _extract_with_structured_method(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using pdfplumber's structured extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                for page_idx, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_idx + 1} with structured method")
                    
                    # Try with different table extraction settings
                    settings_variations = [
                        {},  # Default settings
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                        # More aggressive settings for harder-to-detect tables
                        {
                            "vertical_strategy": "text", 
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 3
                        }
                    ]
                    
                    for settings in settings_variations:
                        try:
                            extracted_tables = page.extract_tables(table_settings=settings)
                            
                            for raw_table in extracted_tables:
                                if raw_table and len(raw_table) > 1:  # Need at least header + data row
                                    # Convert the raw table to DataFrame
                                    df = self._convert_raw_table_to_df(raw_table)
                                    if not df.empty and df.shape[1] >= 2:
                                        tables.append(df)
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error with settings {settings}: {str(e)}")
                            continue
                    
                    # If no tables found with standard settings, try to find invoice table regions
                    if not tables:
                        # Get all text and analyze for table-like structures
                        text = page.extract_text()
                        if text:
                            text_tables = self._extract_tables_from_text(text)
                            if text_tables:
                                tables.extend(text_tables)
        
        except Exception as e:
            logger.error(f"Error in structured extraction: {str(e)}")
        
        return tables
    
    def _extract_with_layout_method(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables by analyzing text layout and positioning.
        Effective for tables without explicit borders.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                for page_idx, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_idx + 1} with layout method")
                    
                    # Extract words with their positions
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    
                    if not words:
                        continue
                    
                    # Group words by y-position (lines)
                    lines = {}
                    for word in words:
                        # Round y-position to group by approximate rows
                        y_key = round(word['top'])
                        if y_key not in lines:
                            lines[y_key] = []
                        lines[y_key].append(word)
                    
                    # Sort lines by y-position
                    sorted_lines = []
                    for y in sorted(lines.keys()):
                        # Sort words in line by x-position
                        sorted_words = sorted(lines[y], key=lambda w: w['x0'])
                        sorted_lines.append(sorted_words)
                    
                    # Look for potential header lines (containing common invoice column names)
                    table_start_indices = []
                    for idx, line in enumerate(sorted_lines):
                        line_text = ' '.join(w['text'] for w in line).lower()
                        
                        # Count how many header patterns match
                        header_matches = sum(1 for pattern in self.table_header_patterns 
                                          if re.search(pattern, line_text))
                        
                        # If multiple header patterns match, this is likely a table header
                        if header_matches >= 2:
                            table_start_indices.append(idx)
                    
                    # For each potential header, extract table
                    for start_idx in table_start_indices:
                        try:
                            # Extract header for column positions
                            header_line = sorted_lines[start_idx]
                            header_positions = [word['x0'] for word in header_line]
                            header_text = [word['text'] for word in header_line]
                            
                            # Extract table rows
                            table_data = [header_text]
                            end_idx = start_idx + 1
                            
                            # Continue until end of table indicators found
                            for i in range(start_idx + 1, len(sorted_lines)):
                                line = sorted_lines[i]
                                if not line:
                                    continue
                                
                                line_text = ' '.join(w['text'] for w in line).lower()
                                
                                # Check if this is the end of the table
                                if any(re.search(pattern, line_text) for pattern in self.table_end_patterns):
                                    # Include this line (often contains totals) and stop
                                    row_data = self._map_words_to_columns(line, header_positions)
                                    table_data.append(row_data)
                                    end_idx = i + 1
                                    break
                                
                                # Map words to columns based on x-position
                                row_data = self._map_words_to_columns(line, header_positions)
                                table_data.append(row_data)
                                end_idx = i + 1
                            
                            # Convert to DataFrame
                            if len(table_data) > 1:
                                # Clean up header
                                header = self._clean_header(table_data[0])
                                data = table_data[1:]
                                
                                # Handle varying row lengths
                                max_cols = max(len(row) for row in data)
                                max_cols = max(max_cols, len(header))
                                
                                # Pad header if needed
                                if len(header) < max_cols:
                                    header.extend([f'Column{i+1}' for i in range(len(header), max_cols)])
                                
                                # Pad data rows
                                for i in range(len(data)):
                                    if len(data[i]) < max_cols:
                                        data[i].extend([''] * (max_cols - len(data[i])))
                                
                                # Create DataFrame
                                df = pd.DataFrame(data, columns=header[:max_cols])
                                if not df.empty and df.shape[1] >= 2:
                                    tables.append(df)
                        
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"Error processing table starting at line {start_idx}: {str(e)}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in layout extraction: {str(e)}")
        
        return tables
    
    def _extract_with_ocr_method(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using OCR for scanned or image-based PDFs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)
            
            for page_idx, img in enumerate(images):
                logger.debug(f"Processing page {page_idx + 1} with OCR method")
                
                # Convert PIL image to OpenCV format
                img_np = np.array(img)
                img_cv = img_np[:, :, ::-1].copy()  # RGB to BGR
                
                # Preprocess image for better OCR
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
                
                # Try to detect table structure using line detection
                horizontal = self._detect_lines(thresh, 'horizontal')
                vertical = self._detect_lines(thresh, 'vertical')
                
                # Combine to get table grid
                table_grid = cv2.bitwise_or(horizontal, vertical)
                
                # Find contours to identify table cells
                contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
                
                # If we found table structure, use it to guide extraction
                if contours and len(contours) > 5:
                    # Get bounding rectangles for each contour
                    rects = [cv2.boundingRect(c) for c in contours]
                    
                    # Group rects by y-coordinate to identify rows
                    rows = self._group_rects_by_rows(rects)
                    
                    # Extract text from each cell using OCR
                    table_data = []
                    for row in rows:
                        row_data = []
                        for x, y, w, h in row:
                            # Extract cell image
                            cell_img = img_np[y:y+h, x:x+w]
                            # Perform OCR on cell
                            cell_text = pytesseract.image_to_string(cell_img).strip()
                            row_data.append(cell_text)
                        
                        if row_data:
                            table_data.append(row_data)
                    
                    # Convert to DataFrame if we have data
                    if len(table_data) > 1:
                        # Identify header row
                        header_idx = self._identify_header_row(table_data)
                        
                        if header_idx >= 0:
                            header = self._clean_header(table_data[header_idx])
                            data = table_data[header_idx+1:]
                        else:
                            header = self._clean_header(table_data[0])
                            data = table_data[1:]
                        
                        # Handle varying row lengths
                        max_cols = max(len(row) for row in data)
                        max_cols = max(max_cols, len(header))
                        
                        # Pad header if needed
                        if len(header) < max_cols:
                            header.extend([f'Column{i+1}' for i in range(len(header), max_cols)])
                        
                        # Pad data rows
                        for i in range(len(data)):
                            if len(data[i]) < max_cols:
                                data[i].extend([''] * (max_cols - len(data[i])))
                        
                        # Create DataFrame
                        df = pd.DataFrame(data, columns=header[:max_cols])
                        if not df.empty and df.shape[1] >= 2:
                            tables.append(df)
                
                # If no table structure detected, run OCR on the entire image
                if not contours or len(contours) <= 5:
                    # Run OCR on the entire image
                    ocr_text = pytesseract.image_to_string(img)
                    
                    # Extract tables from text
                    text_tables = self._extract_tables_from_text(ocr_text)
                    if text_tables:
                        tables.extend(text_tables)
        
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
        
        return tables
    
    def _extract_with_text_patterns(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables by analyzing text patterns (for challenging PDFs).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        try:
            # Extract all text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n\n"
            
            # Process the text to find tables
            text_tables = self._extract_tables_from_text(all_text)
            if text_tables:
                tables.extend(text_tables)
        
        except Exception as e:
            logger.error(f"Error in text pattern extraction: {str(e)}")
        
        return tables
    
    def _extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """
        Extract tables from raw text using pattern analysis.
        
        Args:
            text: Raw text content
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        # Split into lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Find potential header lines (containing multiple header patterns)
        for idx, line in enumerate(lines):
            if idx >= len(lines) - 2:  # Need at least one data row
                continue
                
            line_lower = line.lower()
            
            # Count matches with header patterns
            header_matches = sum(1 for pattern in self.table_header_patterns 
                              if re.search(pattern, line_lower))
            
            # If this looks like a header line
            if header_matches >= 2:
                try:
                    # Determine separator (if any)
                    separator = self._detect_separator(line)
                    
                    # Extract header
                    if separator:
                        header = [col.strip() for col in line.split(separator) if col.strip()]
                    else:
                        # Split by multiple spaces
                        header = re.split(r'\s{2,}', line)
                        header = [col.strip() for col in header if col.strip()]
                    
                    if len(header) < 2:
                        continue
                        
                    # Extract data rows
                    table_data = [header]
                    data_section = False
                    
                    for j in range(idx + 1, len(lines)):
                        data_line = lines[j]
                        
                        # Skip empty lines
                        if not data_line.strip():
                            continue
                        
                        # Check if this is the end of the table
                        if any(re.search(pattern, data_line.lower()) for pattern in self.table_end_patterns):
                            # Include this line (might be totals) then stop
                            if separator:
                                row_data = [col.strip() for col in data_line.split(separator) if col.strip()]
                            else:
                                row_data = re.split(r'\s{2,}', data_line)
                                row_data = [col.strip() for col in row_data if col.strip()]
                            
                            if row_data:
                                table_data.append(row_data)
                                
                            break
                        
                        # Skip lines that appear to be titles, headers or footers
                        if len(data_line) < 10 or data_line.isupper() or data_line.startswith('*'):
                            continue
                            
                        # Extract data using same separator as header
                        if separator:
                            row_data = [col.strip() for col in data_line.split(separator) if col.strip()]
                        else:
                            row_data = re.split(r'\s{2,}', data_line)
                            row_data = [col.strip() for col in row_data if col.strip()]
                        
                        # Skip rows that don't look like data
                        if not row_data or len(row_data) < 2:
                            continue
                            
                        table_data.append(row_data)
                        data_section = True
                    
                    # If we found data rows, convert to DataFrame
                    if data_section and len(table_data) > 1:
                        # Normalize row lengths
                        max_cols = max(len(row) for row in table_data)
                        
                        # Pad rows as needed
                        for i in range(len(table_data)):
                            if len(table_data[i]) < max_cols:
                                table_data[i].extend([''] * (max_cols - len(table_data[i])))
                        
                        # Clean up header
                        header = self._clean_header(table_data[0])
                        
                        # Create DataFrame
                        df = pd.DataFrame(table_data[1:], columns=header)
                        if not df.empty and df.shape[1] >= 2:
                            tables.append(df)
                
                except Exception as e:
                    if self.debug:
                        logger.debug(f"Error extracting table from text at line {idx}: {str(e)}")
                    continue
        
        return tables
    
    def _convert_raw_table_to_df(self, raw_table: List[List[str]]) -> pd.DataFrame:
        """
        Convert a raw table (list of lists) to a DataFrame.
        
        Args:
            raw_table: Raw table data as list of lists
            
        Returns:
            Converted DataFrame
        """
        if not raw_table or len(raw_table) < 2:
            return pd.DataFrame()
        
        # Identify header row (might not be the first row)
        header_idx = self._identify_header_row(raw_table)
        
        if header_idx >= 0:
            header = self._clean_header(raw_table[header_idx])
            data = raw_table[header_idx+1:]
        else:
            header = self._clean_header(raw_table[0])
            data = raw_table[1:]
        
        # Create DataFrame
        try:
            df = pd.DataFrame(data, columns=header)
            return df
        except Exception as e:
            if self.debug:
                logger.debug(f"Error converting raw table to DataFrame: {str(e)}")
            
            # Handle case where header and data rows have different lengths
            max_cols = max(len(row) for row in data)
            max_cols = max(max_cols, len(header))
            
            # Pad header if needed
            if len(header) < max_cols:
                header.extend([f'Column{i+1}' for i in range(len(header), max_cols)])
            
            # Pad data rows
            for i in range(len(data)):
                if len(data[i]) < max_cols:
                    data[i].extend([''] * (max_cols - len(data[i])))
            
            return pd.DataFrame(data, columns=header[:max_cols])
    
    def _identify_header_row(self, table_data: List[List[str]]) -> int:
        """
        Identify which row is most likely to be the header row.
        
        Args:
            table_data: Table data as list of lists
            
        Returns:
            Index of the header row, or -1 if not found
        """
        if not table_data:
            return -1
        
        max_score = -1
        header_idx = -1
        
        for idx, row in enumerate(table_data):
            if not row:
                continue
            
            # Convert to string for pattern matching
            row_text = ' '.join(str(cell).lower() for cell in row if cell)
            
            # Count matches with header patterns
            pattern_matches = sum(1 for pattern in self.table_header_patterns 
                               if re.search(pattern, row_text))
            
            # Headers often have shorter cells and proper capitalization
            word_count = sum(len(str(cell).split()) for cell in row if cell)
            capital_chars = sum(1 for cell in row if isinstance(cell, str) and any(c.isupper() for c in cell))
            
            # Headers rarely contain monetary values
            money_values = sum(1 for cell in row if isinstance(cell, str) and re.search(self.money_pattern, cell))
            
            # Calculate score
            score = pattern_matches * 3 - word_count / 10 + capital_chars - money_values * 2
            
            if score > max_score:
                max_score = score
                header_idx = idx
        
        # Return only if score is reasonable
        return header_idx if max_score > 0 else 0  # Default to first row
    
    def _clean_header(self, header: List[str]) -> List[str]:
        """
        Clean and normalize header column names.
        
        Args:
            header: Raw header values
            
        Returns:
            Cleaned header
        """
        clean_headers = []
        for col in header:
            # Convert to string and clean
            col_text = str(col).strip()
            
            # Remove typical noise
            col_text = re.sub(r'^[_\-:#.*\s]+|[_\-:#.*\s]+$', '', col_text)
            
            # Normalize whitespace
            col_text = re.sub(r'\s+', ' ', col_text)
            
            # Use placeholder for empty column names
            if not col_text:
                col_text = "Column"
                
            clean_headers.append(col_text)
        
        # Handle duplicate column names
        unique_headers = []
        seen = set()
        
        for idx, col in enumerate(clean_headers):
            if col in seen:
                # Add a unique identifier
                col_count = sum(1 for i in range(idx) if clean_headers[i] == col) + 1
                col = f"{col}_{col_count}"
            
            unique_headers.append(col)
            seen.add(col)
        
        return unique_headers
    
    def _map_words_to_columns(self, words: List[Dict], header_positions: List[float]) -> List[str]:
        """
        Map words to columns based on header positions.
        
        Args:
            words: List of word dictionaries with position info
            header_positions: X-coordinates of header columns
            
        Returns:
            List of values mapped to columns
        """
        result = [''] * len(header_positions)
        
        # Sort words by x-position
        sorted_words = sorted(words, key=lambda w: w['x0'])
        
        for word in sorted_words:
            # Find closest header position
            distances = [abs(word['x0'] - pos) for pos in header_positions]
            min_dist = min(distances)
            closest_idx = distances.index(min_dist)
            
            # Only map word if it's reasonably close to a column
            if min_dist <= 100:  # Threshold for mapping to a column
                # Append to that column (in case multiple words map to same column)
                if result[closest_idx]:
                    result[closest_idx] += ' ' + word['text']
                else:
                    result[closest_idx] = word['text']
        
        return result
    
    def _detect_lines(self, img: np.ndarray, direction: str) -> np.ndarray:
        """
        Detect horizontal or vertical lines in an image.
        
        Args:
            img: Input grayscale image
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Image with detected lines
        """
        h, w = img.shape
        
        if direction == 'horizontal':
            kernel_length = w // 30
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        else:  # vertical
            kernel_length = h // 30
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        # Apply morphology to detect lines
        detected = cv2.erode(img, kernel, iterations=2)
        detected = cv2.dilate(detected, kernel, iterations=2)
        
        return detected
    
    def _group_rects_by_rows(self, rects: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """
        Group rectangles by rows based on y-coordinate.
        
        Args:
            rects: List of (x, y, w, h) rectangles
            
        Returns:
            List of rows, each row containing rectangles
        """
        if not rects:
            return []
        
        # Sort by y-coordinate
        sorted_rects = sorted(rects, key=lambda r: r[1])
        
        # Group by rows (similar y-coordinate)
        rows = []
        current_row = [sorted_rects[0]]
        threshold = sorted_rects[0][3] * 0.7  # 70% of height as threshold
        
        for rect in sorted_rects[1:]:
            # If this rect is on the same line (y-coordinates are close)
            if abs(rect[1] - current_row[0][1]) < threshold:
                current_row.append(rect)
            else:
                # Start a new row
                rows.append(sorted(current_row, key=lambda r: r[0]))  # Sort row by x-coordinate
                current_row = [rect]
                threshold = rect[3] * 0.7  # Update threshold based on new row's height
        
        # Add the last row
        if current_row:
            rows.append(sorted(current_row, key=lambda r: r[0]))
        
        return rows
    
    def _detect_separator(self, line: str) -> Optional[str]:
        """
        Detect the separator used in a table line.
        
        Args:
            line: Text line to analyze
            
        Returns:
            Detected separator or None
        """
        for sep in self.common_separators:
            if sep in line and line.count(sep) >= 2:
                return sep
        
        # Check if the line uses consistent spacing
        words = line.split()
        if len(words) >= 3:
            spaces = []
            start = 0
            for word in words:
                pos = line.find(word, start)
                if pos > start:
                    spaces.append(pos - start)
                start = pos + len(word)
            
            # If we have consistent space widths, use them as separators
            if spaces and max(spaces) > 2 * min(spaces):
                return None  # Use regex splitting for space-delimited tables
        
        return None
    
    def _select_best_invoice_table(self, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Select the most likely invoice table from candidates.
        
        Args:
            tables: List of candidate tables
            
        Returns:
            Best table or None if no good candidate
        """
        if not tables:
            return None
        
        # Score each table
        scored_tables = []
        
        for table in tables:
            if table.empty:
                continue
                
            score = 0
            
            # 1. More rows is generally better (up to a point)
            num_rows = len(table)
            if num_rows >= 10:
                score += 5
            elif num_rows >= 5:
                score += 3
            elif num_rows >= 2:
                score += 1
            
            # 2. Number of columns (invoice tables typically have 4-7 columns)
            num_cols = len(table.columns)
            if 4 <= num_cols <= 7:
                score += 4
            elif 3 <= num_cols <= 9:
                score += 2
            else:
                score += 1
            
            # 3. Presence of key columns
            column_names = ' '.join(str(col).lower() for col in table.columns)
            
            # Check for item/description column
            if re.search(r'(?i)\b(item|description|product|article|service)\b', column_names):
                score += 3
            
            # Check for quantity column
            if re.search(r'(?i)\b(qty|quantity|units|pcs|count)\b', column_names):
                score += 2
            
            # Check for price column
            if re.search(r'(?i)\b(price|rate|unit[\s-]?price|cost)\b', column_names):
                score += 2
            
            # Check for amount column
            if re.search(r'(?i)\b(amount|total|line[\s-]?amount|extended|value)\b', column_names):
                score += 3
            
            # 4. Presence of monetary values (essential for invoice tables)
            money_cells = 0
            numeric_cells = 0
            
            for _, row in table.iterrows():
                for val in row:
                    if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r'\d+(\.\d+)?', val)):
                        numeric_cells += 1
                        if isinstance(val, str) and re.search(self.money_pattern, val):
                            money_cells += 1
            
            if money_cells > 0:
                score += min(money_cells, 5)  # Cap at 5 points
            
            if numeric_cells >= num_rows:
                score += 2
            
            # 5. Row consistency (good tables have consistent data types per column)
            consistent_cols = 0
            for col in table.columns:
                col_values = table[col].dropna()
                
                if len(col_values) < 2:
                    continue
                
                # Check for numeric consistency
                numeric_count = sum(1 for val in col_values 
                                  if isinstance(val, (int, float)) or 
                                  (isinstance(val, str) and re.search(r'^[\d,.]+$', val)))
                
                # If mostly numeric or mostly text, it's consistent
                if numeric_count > len(col_values) * 0.8 or numeric_count < len(col_values) * 0.2:
                    consistent_cols += 1
            
            if consistent_cols > len(table.columns) * 0.5:
                score += 3
            
            scored_tables.append((table, score))
            
            if self.debug:
                logger.debug(f"Table with {num_rows} rows, {num_cols} columns, got score: {score}")
        
        # Sort by score (highest first)
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring table, or None if no table has a minimum score
        if scored_tables and scored_tables[0][1] >= 5:
            return scored_tables[0][0]
        elif tables:
            # Fall back to the largest table if all scores are low
            return max(tables, key=len)
        
        return None
    
    def _post_process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize the extracted table.
        
        Args:
            df: Raw extracted table
            
        Returns:
            Processed table
        """
        if df.empty:
            return df
        
        # Copy to avoid modifying the original
        df = df.copy()
        
        # 1. Remove completely empty rows and columns
        df.replace('', np.nan, inplace=True)
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # 2. Normalize column names
        df.columns = self._standardize_column_names(df.columns)
        
        # 3. Clean cell values
        for col in df.columns:
            df[col] = df[col].apply(self._clean_cell_value)
        
        # 4. Convert numeric columns
        for col in df.columns:
            # Skip columns that are clearly not numeric
            if col.lower() in ['description', 'item', 'product', 'service', 'details']:
                continue
                
            # Check if column might be numeric
            values = df[col].dropna()
            numeric_values = 0
            
            for val in values:
                if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r'^[\d,.]+$', val)):
                    numeric_values += 1
            
            # If more than 50% of values look numeric, convert the column
            if numeric_values > len(values) * 0.5:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Drop rows that are likely not part of the table (headers, footers, etc.)
        rows_to_drop = []
        
        for idx, row in df.iterrows():
            # Check if row contains only a single non-null value (likely a header/footer)
            if row.count() == 1 and len(df.columns) > 2:
                rows_to_drop.append(idx)
            
            # Check if row looks like page number or footer
            for val in row:
                if isinstance(val, str) and re.search(r'^page\s+\d+\s+of\s+\d+$', val.lower()):
                    rows_to_drop.append(idx)
                    break
        
        if rows_to_drop:
            df.drop(rows_to_drop, inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _standardize_column_names(self, columns: pd.Index) -> List[str]:
        """
        Standardize column names to common conventions.
        
        Args:
            columns: Original column names
            
        Returns:
            Standardized column names
        """
        std_columns = []
        
        for col in columns:
            col_lower = str(col).lower()
            
            # Check each standard column type
            matched = False
            for std_name, variations in self.standard_columns.items():
                if any(var in col_lower for var in variations):
                    std_columns.append(std_name)
                    matched = True
                    break
            
            # If no match, keep original
            if not matched:
                std_columns.append(str(col))
        
        # Handle duplicates
        unique_columns = []
        seen = set()
        
        for col in std_columns:
            if col in seen:
                count = 1
                while f"{col}_{count}" in seen:
                    count += 1
                unique_columns.append(f"{col}_{count}")
            else:
                unique_columns.append(col)
            
            seen.add(unique_columns[-1])
        
        return unique_columns
    
    def _clean_cell_value(self, value: Any) -> Any:
        """
        Clean a cell value.
        
        Args:
            value: Original cell value
            
        Returns:
            Cleaned value
        """
        if not isinstance(value, str):
            return value
        
        # Trim whitespace
        value = value.strip()
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # Remove common noise characters
        value = re.sub(r'^[-_*#:;\s]+|[-_*#:;\s]+$', '', value)
        
        # Try to convert to numeric if it looks like a number
        if re.match(r'^[\d,.]+$', value):
            try:
                # Remove thousand separators
                clean_val = re.sub(r'[,\s]', '', value)
                if '.' in clean_val:
                    return float(clean_val)
                else:
                    return int(clean_val)
            except ValueError:
                pass
        
        # Try to extract monetary value
        money_match = re.search(self.money_pattern, value)
        if money_match:
            try:
                money_str = money_match.group(0)
                # Remove currency symbols and separators
                clean_val = re.sub(r'[£$€₹,\s]', '', money_str)
                return float(clean_val)
            except ValueError:
                pass
        
        return value
    
    def _is_table_valid(self, table: pd.DataFrame) -> bool:
        """
        Check if a table is valid/useful.
        
        Args:
            table: Table to validate
            
        Returns:
            True if table is valid
        """
        if table is None or table.empty:
            return False
        
        # 1. Must have reasonable shape
        if len(table) < 2 or len(table.columns) < 2:
            return False
        
        # 2. Should have some numeric values
        numeric_cells = 0
        for _, row in table.iterrows():
            for val in row:
                if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r'\d+(\.\d+)?', val)):
                    numeric_cells += 1
        
        if numeric_cells < 5:  # Need at least 5 numeric cells
            return False
        
        # 3. Should have relevant column names
        col_text = ' '.join(str(col).lower() for col in table.columns)
        header_patterns_matched = sum(1 for pattern in self.table_header_patterns 
                                    if re.search(pattern, col_text))
        
        if header_patterns_matched < 1:  # Need at least one relevant column header
            return False
        
        return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract invoice tables from PDF files')
    parser.add_argument('pdf_path', help='Path to the invoice PDF file')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: same as input with .csv extension)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    output_path = args.output
    if not output_path:
        output_path = os.path.splitext(args.pdf_path)[0] + '.csv'
    
    # Initialize extractor
    extractor = AdvancedInvoiceExtractor(debug=args.debug)
    
    try:
        # Process the invoice
        result = extractor.extract_from_pdf(args.pdf_path, output_path)
        
        if result.empty:
            logger.error("Failed to extract invoice table")
            return 1
        
        logger.info(f"Successfully extracted invoice table to: {output_path}")
        
        # Print preview in debug mode
        if args.debug:
            print("\nTable preview:")
            print(result.head())
            print(f"\nTable shape: {result.shape}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())