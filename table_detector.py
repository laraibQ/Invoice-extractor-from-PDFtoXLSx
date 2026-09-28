import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableDetector:
    """
    Class to detect tables in document images using computer vision techniques.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the table detector.
        
        Args:
            debug: Enable debug mode with visualization
        """
        self.debug = debug
        logger.info("TableDetector initialized")
    
    def detect_tables(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect table regions in an image using multiple methods.
        
        Args:
            img: Image as numpy array (BGR format)
            
        Returns:
            List of table regions as (x, y, width, height) tuples
        """
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Get image dimensions
        h, w = gray.shape[:2]
        
        # Try multiple methods and merge results
        table_regions = []
        
        # Method 1: Line detection
        line_tables = self._detect_tables_with_lines(gray)
        if line_tables:
            table_regions.extend(line_tables)
        
        # Method 2: Grid detection
        grid_tables = self._detect_tables_with_grid(gray)
        if grid_tables:
            table_regions.extend(grid_tables)
        
        # Method 3: Layout analysis
        layout_tables = self._detect_tables_with_layout_analysis(gray)
        if layout_tables:
            table_regions.extend(layout_tables)
        
        # Method 4: Text block analysis
        text_tables = self._detect_tables_with_text_blocks(gray)
        if text_tables:
            table_regions.extend(text_tables)
        
        # Merge overlapping regions
        table_regions = self._merge_overlapping_regions(table_regions)
        
        # Filter regions by size
        min_width = int(w * 0.1)  # At least 10% of image width
        min_height = int(h * 0.05)  # At least 5% of image height
        
        filtered_regions = []
        for region in table_regions:
            x, y, rw, rh = region
            if rw >= min_width and rh >= min_height:
                # Ensure region is within image bounds
                x = max(0, x)
                y = max(0, y)
                rw = min(w - x, rw)
                rh = min(h - y, rh)
                
                filtered_regions.append((x, y, rw, rh))
        
        logger.debug(f"Detected {len(filtered_regions)} table regions")
        return filtered_regions
    
    def detect_table_structure(self, img: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Detect the row and column structure of a table.
        
        Args:
            img: Image as numpy array (BGR format)
            
        Returns:
            Tuple of (row_positions, column_positions) as normalized coordinates (0-1)
        """
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Get image dimensions
        h, w = gray.shape[:2]
        
        # Detect horizontal and vertical lines using multiple approaches
        
        # Method 1: Morphological operations
        rows, cols = self._detect_lines_with_morphology(gray)
        
        # If no lines detected, try other methods
        if not rows or not cols:
            # Method 2: Hough Transform
            rows, cols = self._detect_lines_with_hough(gray)
        
        # If still no lines, try edge-based approach
        if not rows or not cols:
            # Method 3: Edge-based
            rows, cols = self._detect_lines_with_edges(gray)
        
        # Normalize coordinates to 0-1 range
        norm_rows = [y / h for y in rows]
        norm_cols = [x / w for x in cols]
        
        logger.debug(f"Detected {len(norm_rows)-1} rows and {len(norm_cols)-1} columns")
        return norm_rows, norm_cols
    
    def _detect_tables_with_lines(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables by looking for horizontal and vertical lines.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of table regions as (x, y, width, height) tuples
        """
        try:
            # Create binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Define kernels for horizontal and vertical lines
            h, w = gray.shape
            horizontal_kernel_size = max(1, int(w * 0.05))  # 5% of width
            vertical_kernel_size = max(1, int(h * 0.05))    # 5% of height
            
            # Horizontal lines kernel
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
            # Vertical lines kernel
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
            
            # Combine horizontal and vertical lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            
            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            table_regions = []
            min_area = gray.shape[0] * gray.shape[1] * 0.01  # At least 1% of the image
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter out very wide or very tall rectangles (likely not tables)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:
                        table_regions.append((x, y, w, h))
            
            return table_regions
        
        except Exception as e:
            logger.error(f"Error in line-based table detection: {str(e)}")
            return []
    
    def _detect_tables_with_grid(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables by looking for grid patterns.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of table regions as (x, y, width, height) tuples
        """
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (descending)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Get image dimensions
            h, w = gray.shape
            
            # Find grid patterns (many small rectangles arranged in a grid)
            rectangles = []
            min_rect_area = h * w * 0.0005  # Minimum cell size
            max_rect_area = h * w * 0.1     # Maximum cell size
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_rect_area < area < max_rect_area:
                    x, y, rw, rh = cv2.boundingRect(contour)
                    aspect_ratio = rw / rh if rh > 0 else 0
                    # Check if it's reasonably rectangular
                    if 0.2 < aspect_ratio < 5:
                        rectangles.append((x, y, rw, rh))
            
            # Group rectangles that form a grid
            table_regions = []
            
            if rectangles:
                # Simple clustering approach: if many rectangles are aligned, they form a table
                # Group rectangles by y-coordinate proximity (for rows)
                y_coords = [rect[1] for rect in rectangles]
                y_coords.sort()
                
                # Find row clusters
                row_clusters = []
                current_cluster = [y_coords[0]]
                
                for i in range(1, len(y_coords)):
                    if y_coords[i] - y_coords[i-1] < h * 0.03:  # 3% of image height
                        current_cluster.append(y_coords[i])
                    else:
                        if len(current_cluster) >= 3:  # At least 3 cells in a row
                            row_clusters.append(current_cluster)
                        current_cluster = [y_coords[i]]
                
                if len(current_cluster) >= 3:
                    row_clusters.append(current_cluster)
                
                # If we have multiple row clusters, likely a table
                if len(row_clusters) >= 3:
                    # Find the bounding box of all the rectangles
                    min_x = min(rect[0] for rect in rectangles)
                    min_y = min(rect[1] for rect in rectangles)
                    max_x = max(rect[0] + rect[2] for rect in rectangles)
                    max_y = max(rect[1] + rect[3] for rect in rectangles)
                    
                    table_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
            return table_regions
        
        except Exception as e:
            logger.error(f"Error in grid-based table detection: {str(e)}")
            return []
    
    def _detect_tables_with_layout_analysis(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables by analyzing the document layout.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of table regions as (x, y, width, height) tuples
        """
        try:
            # Apply bilateral filtering to reduce noise while preserving edges
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Dilate edges to connect nearby lines
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # For each contour, count how many child contours it has
            # Tables often have a hierarchical structure (table -> rows -> cells)
            table_candidates = []
            
            if hierarchy is not None and len(hierarchy) > 0:
                hierarchy = hierarchy[0]
                
                for i, contour in enumerate(contours):
                    # Check if contour has children
                    if hierarchy[i][2] != -1:  # Has at least one child
                        # Count children
                        child_count = 0
                        child_idx = hierarchy[i][2]
                        
                        while child_idx != -1:
                            child_count += 1
                            child_idx = hierarchy[child_idx][0]  # Next sibling
                        
                        # If many children, likely a table
                        if child_count >= 5:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h if h > 0 else 0
                            
                            # Filter by aspect ratio
                            if 0.2 < aspect_ratio < 5:
                                table_candidates.append((x, y, w, h))
            
            return table_candidates
        
        except Exception as e:
            logger.error(f"Error in layout analysis for table detection: {str(e)}")
            return []
    
    def _detect_tables_with_text_blocks(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables by analyzing text block arrangements.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of table regions as (x, y, width, height) tuples
        """
        try:
            # Use MSER to detect text regions
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Convert regions to bounding boxes
            bboxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                # Filter very small regions
                if w > 5 and h > 5:
                    bboxes.append((x, y, w, h))
            
            # No text regions found
            if not bboxes:
                return []
            
            # Cluster text regions by their y-coordinates (to find rows)
            y_coords = [(bbox[1] + bbox[3] // 2) for bbox in bboxes]  # midpoint y
            y_coords = np.array(y_coords).reshape(-1, 1)
            
            # Determine optimal clustering parameters
            h = gray.shape[0]
            y_threshold = h * 0.01  # 1% of image height
            
            # Group boxes into rows
            rows = []
            current_row = [bboxes[0]]
            current_y = y_coords[0][0]
            
            for i in range(1, len(bboxes)):
                y = y_coords[i][0]
                
                # If y-coordinate is similar, add to current row
                if abs(y - current_y) < y_threshold:
                    current_row.append(bboxes[i])
                else:
                    # Start a new row
                    if len(current_row) >= 3:  # Require at least 3 text blocks per row
                        rows.append(current_row)
                    current_row = [bboxes[i]]
                    current_y = y
            
            # Add the last row if it has enough elements
            if len(current_row) >= 3:
                rows.append(current_row)
            
            # If we have multiple rows with similar x-coordinates, likely a table
            if len(rows) >= 3:
                # Check if the rows have columns that align
                # Get all x-coordinates from the first row
                if rows[0]:
                    x_coords = [(box[0], box[0] + box[2]) for box in rows[0]]  # (left, right)
                    
                    # Check if boxes in other rows are aligned with these coordinates
                    aligned_rows = 1  # First row is aligned by definition
                    
                    for row in rows[1:]:
                        aligned_boxes = 0
                        for box in row:
                            # Check if this box aligns with any column from the first row
                            box_left, box_right = box[0], box[0] + box[2]
                            
                            for x_left, x_right in x_coords:
                                if (abs(box_left - x_left) < x_left * 0.1 or  # 10% tolerance
                                    abs(box_right - x_right) < x_right * 0.1):
                                    aligned_boxes += 1
                                    break
                        
                        # If more than half the boxes align, count this row as aligned
                        if aligned_boxes > len(row) / 2:
                            aligned_rows += 1
                    
                    # If most rows align, it's likely a table
                    if aligned_rows > len(rows) * 0.7:  # 70% of rows should align
                        # Get the bounding box of all text regions in the table
                        all_boxes = [box for row in rows for box in row]
                        
                        min_x = min(box[0] for box in all_boxes)
                        min_y = min(box[1] for box in all_boxes)
                        max_x = max(box[0] + box[2] for box in all_boxes)
                        max_y = max(box[1] + box[3] for box in all_boxes)
                        
                        # Add some padding
                        padding_x = int((max_x - min_x) * 0.05)  # 5% padding
                        padding_y = int((max_y - min_y) * 0.05)
                        
                        min_x = max(0, min_x - padding_x)
                        min_y = max(0, min_y - padding_y)
                        max_x = min(gray.shape[1], max_x + padding_x)
                        max_y = min(gray.shape[0], max_y + padding_y)
                        
                        return [(min_x, min_y, max_x - min_x, max_y - min_y)]
            
            return []
        
        except Exception as e:
            logger.error(f"Error in text block analysis for table detection: {str(e)}")
            return []
    
    def _detect_lines_with_morphology(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect table lines using morphological operations.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (row_positions, column_positions)
        """
        h, w = gray.shape
        
        # Binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Define kernels for horizontal and vertical lines
        horizontal_kernel_size = max(1, int(w * 0.3))  # 30% of width
        vertical_kernel_size = max(1, int(h * 0.3))    # 30% of height
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract row positions from horizontal lines
        rows = set()
        for contour in horizontal_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Add the top edge of the line
            rows.add(y)
            # Add the bottom edge of the line
            rows.add(y + h)
        
        # Extract column positions from vertical lines
        cols = set()
        for contour in vertical_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Add the left edge of the line
            cols.add(x)
            # Add the right edge of the line
            cols.add(x + w)
        
        # Add image boundaries
        rows.add(0)
        rows.add(h)
        cols.add(0)
        cols.add(w)
        
        # Sort positions
        rows = sorted(list(rows))
        cols = sorted(list(cols))
        
        return rows, cols
    
    def _detect_lines_with_hough(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect table lines using Hough transform.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (row_positions, column_positions)
        """
        h, w = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=w*0.2, maxLineGap=20)
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is horizontal or vertical
                if abs(y2 - y1) < h * 0.01:  # Horizontal line (< 1% of height)
                    horizontal_lines.append((min(y1, y2), max(y1, y2)))
                elif abs(x2 - x1) < w * 0.01:  # Vertical line (< 1% of width)
                    vertical_lines.append((min(x1, x2), max(x1, x2)))
        
        # Group lines that are close to each other
        # For horizontal lines (rows)
        rows = set()
        for y1, y2 in horizontal_lines:
            # Use average position
            y = (y1 + y2) // 2
            rows.add(y)
        
        # For vertical lines (columns)
        cols = set()
        for x1, x2 in vertical_lines:
            # Use average position
            x = (x1 + x2) // 2
            cols.add(x)
        
        # Add image boundaries
        rows.add(0)
        rows.add(h)
        cols.add(0)
        cols.add(w)
        
        # Sort positions
        rows = sorted(list(rows))
        cols = sorted(list(cols))
        
        return rows, cols
    
    def _detect_lines_with_edges(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect table lines using edge detection and projection profiles.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (row_positions, column_positions)
        """
        h, w = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Horizontal and vertical projection profiles
        h_profile = np.sum(edges, axis=1)
        v_profile = np.sum(edges, axis=0)
        
        # Find peaks in profiles (high edge density)
        h_peaks = []
        v_peaks = []
        
        # For horizontal profile (rows)
        threshold_h = np.mean(h_profile) * 1.5
        for i in range(1, h-1):
            if h_profile[i] > threshold_h and h_profile[i] > h_profile[i-1] and h_profile[i] > h_profile[i+1]:
                h_peaks.append(i)
        
        # For vertical profile (columns)
        threshold_v = np.mean(v_profile) * 1.5
        for i in range(1, w-1):
            if v_profile[i] > threshold_v and v_profile[i] > v_profile[i-1] and v_profile[i] > v_profile[i+1]:
                v_peaks.append(i)
        
        # Add image boundaries
        rows = [0] + h_peaks + [h]
        cols = [0] + v_peaks + [w]
        
        # Sort positions
        rows.sort()
        cols.sort()
        
        return rows, cols
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping table regions.
        
        Args:
            regions: List of (x, y, width, height) regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return []
        
        # Sort by x-coordinate
        regions.sort(key=lambda r: r[0])
        
        merged = []
        current = regions[0]
        
        for region in regions[1:]:
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = region
            
            # Check if regions overlap
            if (x1 <= x2 + w2 and x2 <= x1 + w1 and
                y1 <= y2 + h2 and y2 <= y1 + h1):
                # Merge regions
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                current = (x, y, w, h)
            else:
                merged.append(current)
                current = region
        
        merged.append(current)
        return merged
