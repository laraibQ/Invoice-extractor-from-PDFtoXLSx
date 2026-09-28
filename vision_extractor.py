#!/usr/bin/env python3
"""
Vision-based invoice table extractor.

Uses computer vision, layout analysis, and OCR with post-processing
to pull tabular line items from PDF invoices.
"""

import os
import re
import sys
import logging
import argparse
import tempfile
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from pdf2image import convert_from_path
import pdfplumber
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionInvoiceExtractor:
    """
    Invoice table extraction driven by computer vision and layout analysis.
    Tries structured PDF parsing first, then region detection and OCR.
    """
    
    def __init__(self, tesseract_path=None, debug=False):
        """
        Initialize the vision-based invoice extractor.
        
        Args:
            tesseract_path: Path to tesseract executable
            debug: Enable debug mode with detailed logging and visualizations
        """
        self.debug = debug
        self.confidence_threshold = 0.7  # Confidence threshold for detection
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Enhanced OCR configuration for better accuracy
        self.custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:%$€£-+/()*&# "'
        
        # Invoice-specific keywords and patterns
        self.header_keywords = [
            'item', 'description', 'product', 'qty', 'quantity',
            'price', 'unit price', 'amount', 'total', 'subtotal',
            'vat', 'tax', 'discount', 'part', 'sku', 'code'
        ]
        
        # Patterns for table structure detection
        self.table_header_patterns = [
            r'(?i)item\s*(?:no|number|#|code|sku)?',
            r'(?i)(?:item|product|service)\s*description',
            r'(?i)qty|quantity|units|pcs',
            r'(?i)unit\s*price|price|rate|cost|each',
            r'(?i)amount|total|(?:sub)?total|sum|value',
            r'(?i)vat|tax|gst|hst|vat\s*%',
            r'(?i)discount|disc\.|disc',
        ]
        
        # Coordinates for detected tables
        self.detected_tables = []
        
        # Prepare layout analysis helpers
        self._load_models()
    
    def _load_models(self):
        """Initialize OpenCV-based table detection helpers."""
        try:
            self.table_detector_available = True
            logger.info("Using OpenCV-based layout analysis for table detection")
        except Exception as e:
            logger.warning(f"Could not initialize layout helpers: {str(e)}")
            self.table_detector_available = False
    
    def extract_from_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract invoice table data from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the extracted CSV data
            
        Returns:
            DataFrame containing the extracted table
        """
        logger.info(f"Extracting data from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Process the PDF file
        extracted_dataframes = []
        
        # Convert PDF pages to images for layout analysis
        images = self._pdf_to_images(pdf_path)
        
        # Process each page
        for page_idx, img in enumerate(images):
            logger.info(f"Processing page {page_idx + 1}")
            
            # 1. Detect table regions with computer vision
            table_regions = self._detect_table_regions(img, page_idx)
            
            # 2. Extract text from the PDF page using pdfplumber 
            # (for higher accuracy when PDF has text layer)
            with pdfplumber.open(pdf_path) as pdf:
                if page_idx < len(pdf.pages):
                    page = pdf.pages[page_idx]
                    page_text = page.extract_text()
                    
                    # Try standard table extraction first
                    tables = page.extract_tables()
                    if tables and len(tables) > 0:
                        for table in tables:
                            if table and len(table) > 1:  # Need at least two rows
                                df = pd.DataFrame(table[1:], columns=table[0])
                                extracted_dataframes.append(df)
            
            # 3. If standard extraction failed, use detected regions and OCR
            if not extracted_dataframes:
                for region in table_regions:
                    table_img = self._extract_table_region(img, region)
                    table_df = self._process_table_image(table_img)
                    if table_df is not None and not table_df.empty:
                        extracted_dataframes.append(table_df)
            
            # 4. If still no tables, try full page OCR with table structure detection
            if not extracted_dataframes:
                enhanced_img = self._preprocess_image(img)
                text = pytesseract.image_to_string(enhanced_img, config=self.custom_config)
                
                # Try to extract tables based on text structure
                text_tables = self._extract_tables_from_text(text)
                extracted_dataframes.extend(text_tables)
        
        # Process multiple dataframes if found
        if len(extracted_dataframes) > 0:
            # Select the most likely invoice table
            final_df = self._select_best_invoice_table(extracted_dataframes)
            
            # Post-process to clean and standardize
            final_df = self._post_process_table(final_df)
            
            # Save to CSV if output path provided
            if output_path and not final_df.empty:
                final_df.to_csv(output_path, index=False)
                logger.info(f"Table saved to: {output_path}")
            
            return final_df
        
        logger.warning("No tables detected in the PDF")
        return pd.DataFrame()
    
    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images for processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of images as numpy arrays
        """
        try:
            # Convert PDF to high-resolution images
            pil_images = convert_from_path(pdf_path, dpi=300)
            
            # Convert PIL images to OpenCV format
            cv_images = []
            for pil_img in pil_images:
                # Convert to RGB to ensure consistent color space
                img_np = np.array(pil_img.convert('RGB'))
                # Convert RGB to BGR (OpenCV format)
                img_cv = img_np[:, :, ::-1].copy()
                cv_images.append(img_cv)
            
            return cv_images
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return []
    
    def _detect_table_regions(self, img: np.ndarray, page_idx: int) -> List[List[int]]:
        """
        Detect table regions in an image using OpenCV layout analysis.
        
        Args:
            img: Image as numpy array
            page_idx: Page index for debugging
            
        Returns:
            List of regions as [x, y, width, height]
        """
        regions = []
        
        try:
            # 1. Convert to grayscale for processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 9)
            
            # 3. Remove noise
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 4. Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # 5. Combine horizontal and vertical lines to get table structure
            table_structure = cv2.add(horizontal_lines, vertical_lines)
            
            # 6. Find contours of table structure
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 7. Filter contours by size to get tables
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter small areas and ensure reasonable aspect ratio
                if area > (img.shape[0] * img.shape[1]) * 0.01 and 0.1 < w/h < 10:
                    # Expand region slightly to catch borders
                    x_expanded = max(0, x - 10)
                    y_expanded = max(0, y - 10)
                    w_expanded = min(img.shape[1] - x_expanded, w + 20)
                    h_expanded = min(img.shape[0] - y_expanded, h + 20)
                    
                    regions.append([x_expanded, y_expanded, w_expanded, h_expanded])
            
            # 8. If no tables detected with lines, try text-based detection
            if not regions:
                # Detect text lines using Tesseract's layout analysis
                # This helps with tables that don't have visible grid lines
                page_info = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Group text blocks by y-coordinates to identify rows
                y_coords = {}
                for i in range(len(page_info['text'])):
                    if page_info['text'][i].strip():
                        y = page_info['top'][i]
                        # Group with similar y-coordinates (within threshold)
                        y_key = y // 15 * 15  # Group within 15 pixels
                        if y_key not in y_coords:
                            y_coords[y_key] = []
                        y_coords[y_key].append((
                            page_info['left'][i],
                            page_info['width'][i],
                            page_info['text'][i]
                        ))
                
                # Find regions with structured text (aligned blocks)
                if len(y_coords) > 5:  # Need minimum number of lines
                    # Find min/max coordinates
                    all_lines = []
                    for y, line_elements in sorted(y_coords.items()):
                        # Sort elements by x-coordinate
                        line_elements.sort()
                        line_text = ' '.join(text for _, _, text in line_elements)
                        all_lines.append((y, line_text))
                    
                    # Find header row candidates
                    for i, (y, line_text) in enumerate(all_lines):
                        header_matches = sum(1 for pattern in self.table_header_patterns
                                          if re.search(pattern, line_text.lower()))
                        
                        # If line matches multiple patterns, might be a header
                        if header_matches >= 2:
                            # Find table end
                            end_idx = min(i + 25, len(all_lines))  # Look ahead max 25 lines
                            
                            # Determine table boundaries
                            min_x, max_x = float('inf'), 0
                            min_y, max_y = y, 0
                            
                            for j in range(i, end_idx):
                                if j < len(all_lines):
                                    row_y = all_lines[j][0]
                                    elements = y_coords.get(row_y, [])
                                    
                                    if elements:
                                        min_x = min(min_x, min(x for x, _, _ in elements))
                                        max_x = max(max_x, max(x + w for x, w, _ in elements))
                                        max_y = row_y + 20  # Approximate height
                            
                            # Create region if reasonable
                            if max_y > min_y and max_x > min_x:
                                region_w = max_x - min_x + 50  # Add margin
                                region_h = max_y - min_y + 30
                                regions.append([min_x - 20, min_y - 15, region_w, region_h])
            
            # 9. Save detected regions for debugging
            if self.debug and regions:
                debug_img = img.copy()
                for i, (x, y, w, h) in enumerate(regions):
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_img, f"Table {i+1}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                debug_dir = "debug_output"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/page_{page_idx+1}_tables.jpg", debug_img)
            
            # If no regions detected, consider the whole page
            if not regions:
                h, w = img.shape[:2]
                # Exclude page margins
                margin = int(min(h, w) * 0.03)
                regions.append([margin, margin, w - 2*margin, h - 2*margin])
        
        except Exception as e:
            logger.error(f"Error detecting table regions: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # Fallback to whole page
            h, w = img.shape[:2]
            regions.append([0, 0, w, h])
        
        self.detected_tables = regions
        return regions
    
    def _extract_table_region(self, img: np.ndarray, region: List[int]) -> np.ndarray:
        """
        Extract a table region from an image.
        
        Args:
            img: Full page image
            region: Table region as [x, y, width, height]
            
        Returns:
            Cropped and processed table image
        """
        x, y, w, h = region
        table_img = img[y:y+h, x:x+w].copy()
        
        # Apply preprocessing to enhance the table image
        return self._preprocess_image(table_img)
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            img: Input image
            
        Returns:
            Processed image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Convert back to PIL for Tesseract
            pil_img = Image.fromarray(denoised)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            # Convert back to numpy array
            return np.array(pil_img)
        
        except Exception as e:
            logger.warning(f"Error preprocessing image: {str(e)}")
            # Return original grayscale if errors
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
    
    def _process_table_image(self, img: np.ndarray) -> Optional[pd.DataFrame]:
        """
        Process a table image to extract a structured DataFrame.
        
        Args:
            img: Preprocessed table image
            
        Returns:
            DataFrame containing table data or None if extraction fails
        """
        try:
            # 1. Use OCR to extract text with position information
            ocr_data = pytesseract.image_to_data(img, config=self.custom_config, 
                                               output_type=pytesseract.Output.DICT)
            
            # 2. Filter out empty text and low confidence
            filtered_indices = []
            for i in range(len(ocr_data["text"])):
                text = ocr_data["text"][i].strip()
                conf = float(ocr_data["conf"][i])
                
                if text and conf > 30:  # Reasonable confidence threshold
                    filtered_indices.append(i)
            
            if not filtered_indices:
                return None
            
            # 3. Group by line (based on top coordinate)
            lines = {}
            for i in filtered_indices:
                top = ocr_data["top"][i]
                # Group with similar y-coordinates (within threshold)
                y_key = top // 12 * 12  # Group within 12 pixels
                
                if y_key not in lines:
                    lines[y_key] = []
                
                lines[y_key].append({
                    "text": ocr_data["text"][i],
                    "left": ocr_data["left"][i],
                    "width": ocr_data["width"][i],
                    "conf": float(ocr_data["conf"][i])
                })
            
            # 4. Sort lines by y-coordinate
            sorted_lines = []
            for y in sorted(lines.keys()):
                # Sort words in line by x-coordinate
                sorted_words = sorted(lines[y], key=lambda w: w["left"])
                line_text = " ".join(w["text"] for w in sorted_words)
                sorted_lines.append((y, sorted_words, line_text))
            
            # 5. Identify header line
            header_idx = -1
            for idx, (_, _, line_text) in enumerate(sorted_lines):
                header_matches = sum(1 for pattern in self.table_header_patterns
                                  if re.search(pattern, line_text.lower()))
                if header_matches >= 2:
                    header_idx = idx
                    break
            
            # If no header found, try to guess based on position
            if header_idx == -1 and len(sorted_lines) >= 2:
                # Assume first non-empty line is header
                header_idx = 0
            
            if header_idx == -1 or header_idx >= len(sorted_lines) - 1:
                # Need at least a header and one data row
                return None
            
            # 6. Extract header
            _, header_words, _ = sorted_lines[header_idx]
            
            # Find column positions based on header words
            column_positions = []
            for word in header_words:
                center = word["left"] + word["width"] // 2
                column_positions.append((center, word["text"]))
            
            # 7. Process data rows
            table_data = []
            for idx in range(header_idx + 1, len(sorted_lines)):
                y, line_words, _ = sorted_lines[idx]
                
                # Skip empty lines
                if not line_words:
                    continue
                
                # Map words to columns based on position
                row_data = [""] * len(column_positions)
                
                for word in line_words:
                    word_center = word["left"] + word["width"] // 2
                    
                    # Find closest column
                    distances = [abs(word_center - pos[0]) for pos in column_positions]
                    min_dist_idx = distances.index(min(distances))
                    
                    # Append to column content
                    if row_data[min_dist_idx]:
                        row_data[min_dist_idx] += " " + word["text"]
                    else:
                        row_data[min_dist_idx] = word["text"]
                
                # Add non-empty rows to table
                if any(cell.strip() for cell in row_data):
                    table_data.append(row_data)
            
            # 8. Create DataFrame
            if table_data:
                headers = [pos[1] for pos in column_positions]
                # Clean header names
                headers = self._clean_header_names(headers)
                
                df = pd.DataFrame(table_data, columns=headers)
                return df
            
            return None
        
        except Exception as e:
            logger.error(f"Error processing table image: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """
        Extract tables from raw text using pattern analysis.
        
        Args:
            text: Raw OCR text
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        
        try:
            # Split text into lines
            lines = text.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            if not lines:
                return tables
            
            # Look for table patterns
            for idx, line in enumerate(lines):
                if idx >= len(lines) - 2:  # Need at least one data row
                    continue
                
                line_lower = line.lower()
                
                # Check if this looks like a header row
                header_matches = sum(1 for pattern in self.table_header_patterns
                                  if re.search(pattern, line_lower))
                
                if header_matches >= 2:
                    # Try to determine separator
                    separators = ["|", "\t"]
                    separator = None
                    
                    for sep in separators:
                        if sep in line and line.count(sep) >= 2:
                            separator = sep
                            break
                    
                    # Extract header
                    if separator:
                        # Split by separator
                        header = [col.strip() for col in line.split(separator) if col.strip()]
                    else:
                        # Try to split by whitespace
                        header = re.split(r'\s{2,}', line)
                        header = [col.strip() for col in header if col.strip()]
                    
                    if len(header) < 2:
                        continue
                    
                    # Clean header names
                    header = self._clean_header_names(header)
                    
                    # Extract data rows
                    table_data = []
                    for j in range(idx + 1, len(lines)):
                        data_line = lines[j]
                        
                        # Skip empty lines or likely non-data lines
                        if not data_line.strip() or len(data_line) < 5:
                            continue
                        
                        # Extract data using same separator
                        if separator:
                            row_data = [col.strip() for col in data_line.split(separator)]
                        else:
                            # Try to align with header positions
                            row_data = re.split(r'\s{2,}', data_line)
                            row_data = [col.strip() for col in row_data]
                        
                        # Skip row if it doesn't match column count approximately
                        if len(row_data) < len(header) * 0.5 or len(row_data) > len(header) * 1.5:
                            continue
                        
                        # Pad row if needed
                        if len(row_data) < len(header):
                            row_data.extend([''] * (len(header) - len(row_data)))
                        elif len(row_data) > len(header):
                            # If too many columns, combine extras with last valid column
                            extra = ' '.join(row_data[len(header):])
                            row_data = row_data[:len(header) - 1] + [row_data[len(header) - 1] + ' ' + extra]
                        
                        table_data.append(row_data)
                        
                        # Stop if encounter likely end of table
                        if any(re.search(r'(?i)total|subtotal|balance', c) for c in row_data):
                            break
                    
                    # Create DataFrame if we found data
                    if table_data:
                        df = pd.DataFrame(table_data, columns=header)
                        tables.append(df)
                        
                        # Avoid detecting same table twice
                        end_row = j if 'j' in locals() else idx
        
        except Exception as e:
            logger.error(f"Error extracting tables from text: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
        
        return tables
    
    def _clean_header_names(self, headers: List[str]) -> List[str]:
        """
        Clean and normalize header names.
        
        Args:
            headers: Raw header values
            
        Returns:
            Cleaned header names
        """
        cleaned = []
        
        # Standard mappings for common invoice headers
        common_headers = {
            'item': ['item', 'sku', 'code', 'part', 'no.', 'number'],
            'description': ['description', 'desc', 'product', 'service', 'item name', 'particulars'],
            'quantity': ['quantity', 'qty', 'units', 'pcs', 'count'],
            'unit_price': ['price', 'unit price', 'rate', 'cost', 'each'],
            'amount': ['amount', 'total', 'line total', 'extended', 'net', 'value'],
            'vat': ['vat', 'tax', 'gst', 'hst', 'tax rate', 'vat rate'],
            'discount': ['discount', 'disc', 'disc.']
        }
        
        # Clean each header
        for col in headers:
            # Convert to string and clean
            col_str = str(col).strip()
            
            # Remove noise characters
            col_str = re.sub(r'^[_\-:#.*\s]+|[_\-:#.*\s]+$', '', col_str)
            
            # Normalize whitespace
            col_str = re.sub(r'\s+', ' ', col_str)
            
            # Use empty placeholder if needed
            if not col_str:
                col_str = "Column"
            
            # Map to standard name if applicable
            col_lower = col_str.lower()
            matched = False
            
            for std_name, variations in common_headers.items():
                if any(v in col_lower for v in variations):
                    cleaned.append(std_name)
                    matched = True
                    break
            
            if not matched:
                cleaned.append(col_str)
        
        # Handle duplicates
        final_headers = []
        seen = {}
        
        for header in cleaned:
            if header in seen:
                seen[header] += 1
                header = f"{header}_{seen[header]}"
            else:
                seen[header] = 1
            
            final_headers.append(header)
        
        return final_headers
    
    def _select_best_invoice_table(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Select the most likely invoice table from candidates.
        
        Args:
            tables: List of candidate tables
            
        Returns:
            Best table or empty DataFrame if none found
        """
        if not tables:
            return pd.DataFrame()
        
        # Only one table, return it
        if len(tables) == 1:
            return tables[0]
        
        # Score tables based on invoice-specific features
        scored_tables = []
        
        for table in tables:
            if table.empty or len(table) < 2 or len(table.columns) < 2:
                continue
            
            score = 0
            col_text = ' '.join(str(col).lower() for col in table.columns)
            
            # 1. Score based on expected column headers
            if re.search(r'item|sku|code|product', col_text):
                score += 5
            if re.search(r'desc|description', col_text):
                score += 5
            if re.search(r'qty|quantity|units', col_text):
                score += 4
            if re.search(r'price|rate|cost', col_text):
                score += 4
            if re.search(r'amount|total|value', col_text):
                score += 5
            if re.search(r'vat|tax|gst', col_text):
                score += 3
            
            # 2. Score based on numeric values
            # Invoice tables should have numeric columns (amounts, quantities, etc.)
            numeric_cols = 0
            for col in table.columns:
                numeric_values = 0
                try:
                    numeric_values = sum(1 for val in table[col] if isinstance(val, (int, float)) or
                                        (isinstance(val, str) and re.search(r'^[\d,.]+$', val)))
                except:
                    pass
                
                if numeric_values > len(table) * 0.5:
                    numeric_cols += 1
            
            score += min(numeric_cols * 2, 8)  # Cap at 8 points
            
            # 3. Score based on number of rows (prefer tables with more rows)
            row_score = min(len(table) // 2, 5)
            score += row_score
            
            # 4. Score based on currency symbols (very indicative of invoice tables)
            currency_matches = sum(1 for val in table.values.flatten() 
                               if isinstance(val, str) and re.search(r'[$€£¥]', val))
            score += min(currency_matches, 3)
            
            # Add to scored tables
            scored_tables.append((table, score))
        
        # Return highest scored table
        if scored_tables:
            scored_tables.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Selected best table with score {scored_tables[0][1]}")
            return scored_tables[0][0]
        
        # If no tables scored, return the largest one
        return max(tables, key=lambda t: len(t))
    
    def _post_process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the extracted table.
        
        Args:
            df: Raw extracted table
            
        Returns:
            Processed table
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # 1. Remove completely empty rows and columns
        df.replace('', np.nan, inplace=True)
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # 2. Clean cell values
        for col in df.columns:
            df[col] = df[col].apply(self._clean_cell_value)
        
        # 3. Convert numeric columns where appropriate
        for col in df.columns:
            # Skip columns that are clearly text-based
            if col in ['description', 'item', 'product', 'service', 'comments', 'notes']:
                continue
            
            # Check if column has numeric values
            numeric_count = sum(1 for val in df[col].dropna() 
                            if isinstance(val, (int, float)) or 
                            (isinstance(val, str) and re.search(r'^[$€£¥\d,.]+$', val.strip())))
            
            # If mostly numeric, convert to numeric
            if numeric_count > len(df[col].dropna()) * 0.7:
                # Convert, but keep non-convertible as is
                df[col] = pd.to_numeric(df[col].apply(self._normalize_numeric), errors='ignore')
        
        # 4. Try to identify common invoice columns if not already standardized
        self._identify_column_types(df)
        
        # 5. Sort columns to standard order if common columns identified
        self._sort_columns(df)
        
        # 6. Drop duplicated rows
        df = df.drop_duplicates().reset_index(drop=True)
        
        return df
    
    def _clean_cell_value(self, value: Any) -> Any:
        """
        Clean a cell value in the extracted table.
        
        Args:
            value: Original cell value
            
        Returns:
            Cleaned value
        """
        if not isinstance(value, str):
            return value
        
        # Trim whitespace
        value = value.strip()
        
        # Remove noise characters
        value = value.replace('\n', ' ').replace('\r', ' ')
        value = re.sub(r'\s+', ' ', value)
        
        # Clean up common OCR errors
        value = value.replace('|', '1').replace('O', '0')
        
        # Try to convert numbers with currency symbols
        if re.match(r'^[$€£¥]?\s*[\d,.]+$', value):
            try:
                return self._normalize_numeric(value)
            except:
                pass
        
        return value
    
    def _normalize_numeric(self, value: Any) -> Any:
        """
        Normalize numeric and currency values.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized numeric value or original
        """
        if not isinstance(value, str):
            return value
        
        try:
            # Remove currency symbols and whitespace
            clean_val = re.sub(r'[$€£¥\s]', '', value)
            
            # Handle European format (comma as decimal separator)
            if ',' in clean_val and '.' in clean_val:
                # Check which is likely decimal separator
                if clean_val.rindex('.') > clean_val.rindex(','):
                    # US format: remove thousand separators (commas)
                    clean_val = clean_val.replace(',', '')
                else:
                    # European format: convert to US format
                    clean_val = clean_val.replace('.', '').replace(',', '.')
            elif ',' in clean_val and '.' not in clean_val:
                # Might be European format
                clean_val = clean_val.replace(',', '.')
            
            # Convert to float
            return float(clean_val)
        except:
            return value
    
    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identify common invoice column types and rename if needed.
        
        Args:
            df: DataFrame to process
        """
        # Only apply if columns aren't already standardized
        std_columns = ['item', 'description', 'quantity', 'unit_price', 'amount', 'vat', 'discount']
        if all(col in df.columns for col in std_columns):
            return
        
        # Check column contents to identify types
        column_scores = {}
        
        for col in df.columns:
            col_str = str(col).lower()
            column_scores[col] = {
                'item': 0,
                'description': 0, 
                'quantity': 0,
                'unit_price': 0,
                'amount': 0,
                'vat': 0,
                'discount': 0
            }
            
            # Score by name
            if any(word in col_str for word in ['item', 'sku', 'code', 'no', 'part']):
                column_scores[col]['item'] += 10
            
            if any(word in col_str for word in ['desc', 'product', 'service', 'particular']):
                column_scores[col]['description'] += 10
            
            if any(word in col_str for word in ['qty', 'quantity', 'units', 'pcs']):
                column_scores[col]['quantity'] += 10
            
            if any(word in col_str for word in ['price', 'rate', 'unit', 'cost', 'each']):
                column_scores[col]['unit_price'] += 10
            
            if any(word in col_str for word in ['amount', 'total', 'value', 'net']):
                column_scores[col]['amount'] += 10
            
            if any(word in col_str for word in ['vat', 'tax', 'gst', 'hst']):
                column_scores[col]['vat'] += 10
            
            if any(word in col_str for word in ['discount', 'disc']):
                column_scores[col]['discount'] += 10
            
            # Score by content
            try:
                # If mostly short text with numbers, might be item codes
                if sum(1 for val in df[col] if isinstance(val, str) and
                     len(val) < 15 and re.search(r'[A-Z0-9]', str(val))) > len(df) * 0.7:
                    column_scores[col]['item'] += 5
                
                # If mostly long text, likely description
                if sum(1 for val in df[col] if isinstance(val, str) and len(val) > 15) > len(df) * 0.5:
                    column_scores[col]['description'] += 8
                
                # If mostly small whole numbers, likely quantity
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if not numeric_vals.isna().all():
                    if sum(1 for val in numeric_vals.dropna() if val <= 100 and val.is_integer()) > len(numeric_vals.dropna()) * 0.7:
                        column_scores[col]['quantity'] += 8
                    
                    # If moderate numbers, likely unit price
                    avg_val = numeric_vals.mean()
                    if 0.1 < avg_val < 1000:
                        column_scores[col]['unit_price'] += 5
                    
                    # If larger numbers, likely amount
                    if avg_val > 10:
                        column_scores[col]['amount'] += 5
                
                # If contains % symbol or small values, likely VAT/tax
                if sum(1 for val in df[col] if isinstance(val, str) and '%' in val) > 0:
                    column_scores[col]['vat'] += 8
                elif numeric_vals.mean() < 25 and numeric_vals.mean() > 0:
                    column_scores[col]['vat'] += 3
                
                # If contains negative values or % symbol, could be discount
                if sum(1 for val in numeric_vals.dropna() if val < 0) > 0:
                    column_scores[col]['discount'] += 8
            except:
                pass
        
        # Assign types based on scores
        assigned_types = {}
        for col_type in ['item', 'description', 'quantity', 'unit_price', 'amount', 'vat', 'discount']:
            # Get column with highest score for this type
            best_col = max(column_scores.keys(), 
                          key=lambda c: column_scores[c][col_type])
            
            # Only assign if score is reasonable
            if column_scores[best_col][col_type] >= 5 and best_col not in assigned_types.values():
                assigned_types[col_type] = best_col
        
        # Rename columns based on assignments
        rename_map = {old_col: new_name for new_name, old_col in assigned_types.items()}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
    
    def _sort_columns(self, df: pd.DataFrame) -> None:
        """
        Sort columns to standard order if common columns are present.
        
        Args:
            df: DataFrame to sort
        """
        # Define standard order
        standard_order = ['item', 'description', 'quantity', 'unit_price', 'discount', 'vat', 'amount']
        
        # Get columns that match standard order
        present_std_cols = [col for col in standard_order if col in df.columns]
        
        # Get other columns
        other_cols = [col for col in df.columns if col not in standard_order]
        
        # Reorder columns
        if present_std_cols:
            df = df[present_std_cols + other_cols]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Invoice CSV Studio — vision extractor')
    parser.add_argument('pdf_path', help='Path to the invoice PDF file')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: same as input with .csv extension)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode with visualizations')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    output_path = args.output
    if not output_path:
        output_path = os.path.splitext(args.pdf_path)[0] + '.csv'
    
    # Initialize extractor
    extractor = VisionInvoiceExtractor(debug=args.debug)
    
    try:
        # Process the invoice
        logger.info(f"Processing invoice: {args.pdf_path}")
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
        logger.error(f"Error processing invoice: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())