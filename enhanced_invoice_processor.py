#!/usr/bin/env python3
"""
Enhanced Invoice Table Extractor

This script provides advanced table extraction for complex PDF invoices.
It handles multi-page tables, varying formats, and extracts only the relevant
invoice table while filtering out noise.

Usage:
    python enhanced_invoice_processor.py <pdf_path> [--output <output_path>] [--debug]
"""

import os
import sys
import argparse
import logging
from typing import Optional
import pandas as pd
from table_extraction_enhancement import EnhancedTableExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """
    Main class for processing invoice PDFs and extracting tables.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the invoice processor.
        
        Args:
            debug: Enable debug mode with verbose output
        """
        self.debug = debug
        self.table_extractor = EnhancedTableExtractor(debug=debug)
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> bool:
        """
        Process a PDF invoice and extract the main table to CSV.
        
        Args:
            pdf_path: Path to the invoice PDF file
            output_path: Path to save the output CSV (if None, uses the PDF name with .csv extension)
            
        Returns:
            Success status (True if table was found and saved)
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return False
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.splitext(pdf_path)[0] + '.csv'
        
        logger.info(f"Processing invoice: {pdf_path}")
        
        try:
            # Step 1: Extract all potential tables from the PDF
            tables = self.table_extractor.extract_tables_from_pdf(pdf_path)
            
            if not tables:
                logger.warning(f"No tables found in: {pdf_path}")
                return False
            
            # Step 2: Select the most likely invoice table
            invoice_table = self.table_extractor.select_best_invoice_table(tables)
            
            if invoice_table is None or invoice_table.empty:
                logger.warning(f"No valid invoice table found in: {pdf_path}")
                return False
            
            # Step 3: Save the table to CSV
            invoice_table.to_csv(output_path, index=False)
            logger.info(f"Invoice table extracted and saved to: {output_path}")
            
            if self.debug:
                logger.debug(f"Table shape: {invoice_table.shape}")
                logger.debug(f"Table columns: {invoice_table.columns.tolist()}")
                logger.debug(f"Preview of extracted table:\n{invoice_table.head(3)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def process_directory(self, directory_path: str, output_dir: Optional[str] = None) -> tuple:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            output_dir: Directory to save output CSV files (if None, uses same directory)
            
        Returns:
            Tuple of (success_count, total_count)
        """
        if not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return (0, 0)
        
        # Set default output directory if not provided
        if not output_dir:
            output_dir = directory_path
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in: {directory_path}")
            return (0, 0)
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        success_count = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            output_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + '.csv')
            
            if self.process_pdf(pdf_path, output_path):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count} out of {len(pdf_files)} files")
        return (success_count, len(pdf_files))


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Extract tables from invoice PDFs')
    parser.add_argument('path', help='Path to invoice PDF file or directory of PDF files')
    parser.add_argument('-o', '--output', help='Output CSV file or directory (if input is directory)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = InvoiceProcessor(debug=args.debug)
    
    # Process file or directory
    if os.path.isdir(args.path):
        success_count, total_count = processor.process_directory(args.path, args.output)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        logger.info(f"Batch processing complete. Success rate: {success_rate:.1f}%")
        return 0 if success_count > 0 else 1
    else:
        success = processor.process_pdf(args.path, args.output)
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())