import cv2
import numpy as np

# ============================================================================
# FUNCTION 1: Find ball clusters (mask of the balls)
# ============================================================================
def find_ball_clusters(image_path):
    """
    Find metal ball clusters using grayscale intensity and edge detection.
    Metal balls are typically shiny/reflective with high contrast edges.
    
    Args:
        image_path: Path to input image
        
    Returns:
        frame: Original image
        mask: Binary mask of metal ball regions
        contours: List of contours for each cluster
    """
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Could not read image")
        return None, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect bright reflective surfaces (metal balls are often shiny)
    # Threshold for bright regions
    _, bright_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # Method 2: Edge detection (metal balls have sharp circular edges)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to close gaps
    kernel_edge = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
    
    # Method 3: Adaptive thresholding (works well for metal in varying lighting)
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, -2
    )
    
    # Combine masks: bright regions OR strong edges
    combined_mask = cv2.bitwise_or(bright_mask, edges_dilated)
    combined_mask = cv2.bitwise_or(combined_mask, adaptive_mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill holes in detected regions
    contours_temp, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(combined_mask)
    cv2.drawContours(mask, contours_temp, -1, 255, -1)  # Fill contours
    
    # Find final contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area and circularity
    MIN_AREA = 300
    filtered_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            # Check if reasonably circular (optional filter)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Metal balls should be fairly circular
                if circularity > 0.4:  # Relaxed threshold for overlapping balls
                    filtered_contours.append(c)
    
    return frame, mask, filtered_contours


# ============================================================================
# FUNCTION 2: Estimate distance from camera to cluster (depth estimation)
# ============================================================================
def estimate_distance(contour, focal_length=800, real_ball_diameter_cm=2.54):
    """
    Estimate distance to a ball cluster using pinhole camera model.
    
    Args:
        contour: OpenCV contour of the cluster
        focal_length: Camera focal length in pixels (needs calibration)
        real_ball_diameter_cm: Real diameter of metal ball in cm (1 inch = 2.54cm)
        
    Returns:
        distance_cm: Estimated distance in centimeters
        pixel_diameter: Diameter in pixels (for debugging)
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Use the larger dimension as diameter approximation
    pixel_diameter = max(w, h)
    
    # Pinhole camera model: Distance = (Real_Size * Focal_Length) / Pixel_Size
    if pixel_diameter > 0:
        distance_cm = (real_ball_diameter_cm * focal_length) / pixel_diameter
    else:
        distance_cm = 0
    
    return distance_cm, pixel_diameter


# ============================================================================
# FUNCTION 3: Count number of balls per cluster (using YOLOv8 or other model)
# ============================================================================
def count_balls_per_cluster_yolo(frame, contour, model_path='yolov8n.pt'):
    """
    Count balls in a cluster using YOLOv8 object detection.
    
    NOTE: This requires:
    - pip install ultralytics
    - A trained YOLOv8 model for metal balls
    
    Args:
        frame: Original image
        contour: Contour of the cluster region
        model_path: Path to YOLOv8 model weights
        
    Returns:
        ball_count: Number of balls detected in the cluster
        detections: List of detection bounding boxes
    """
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model = YOLO(model_path)
        
        # Get bounding box of cluster
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Crop cluster region
        cluster_roi = frame[y1:y2, x1:x2]
        
        # Run inference
        results = model(cluster_roi, conf=0.5)
        
        # Count detections
        detections = []
        ball_count = 0
        
        for result in results:
            boxes = result.boxes
            ball_count = len(boxes)
            
            for box in boxes:
                # Get box coordinates (relative to ROI)
                x_box, y_box, w_box, h_box = box.xywh[0].cpu().numpy()
                detections.append({
                    'x': int(x1 + x_box - w_box/2),
                    'y': int(y1 + y_box - h_box/2),
                    'w': int(w_box),
                    'h': int(h_box),
                    'conf': float(box.conf[0])
                })
        
        return ball_count, detections
        
    except ImportError:
        print("YOLOv8 not installed. Run: pip install ultralytics")
        return -1, []
    except Exception as e:
        print(f"YOLO inference error: {e}")
        return -1, []


def count_balls_per_cluster_hough(frame, mask, contour):
    """
    Count balls in a cluster using Hough Circle Transform (non-ML alternative).
    Optimized for metal balls with strong circular edges.
    
    Args:
        frame: Original image
        mask: Binary mask of metal ball regions
        contour: Contour of the cluster region
        
    Returns:
        ball_count: Number of circles detected
        circles: List of detected circles (x, y, radius)
    """
    # Get bounding box of cluster
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    # Crop cluster region
    cluster_roi = frame[y1:y2, x1:x2]
    mask_roi = mask[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(cluster_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply mask to focus only on metal ball regions
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_roi)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
    
    # Detect circles using Hough Transform
    # For metal balls, edges are usually very sharp
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,  # Minimum distance between circle centers (adjust for ball size)
        param1=50,   # Canny edge threshold
        param2=25,   # Accumulator threshold (lower = more circles detected)
        minRadius=8,
        maxRadius=80
    )
    
    detected_circles = []
    ball_count = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        ball_count = len(circles[0])
        
        for circle in circles[0]:
            cx, cy, r = circle
            detected_circles.append({
                'x': int(x1 + cx),
                'y': int(y1 + cy),
                'radius': int(r)
            })
    
    return ball_count, detected_circles


def count_balls_per_cluster_blob(frame, mask, contour):
    """
    Count balls using blob detection (alternative non-ML method).
    Works well for metal balls with consistent size and shape.
    
    Args:
        frame: Original image
        mask: Binary mask of metal ball regions
        contour: Contour of the cluster region
        
    Returns:
        ball_count: Number of blobs detected
        blobs: List of detected blob keypoints
    """
    # Get bounding box of cluster
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    # Crop cluster region
    mask_roi = mask[y1:y2, x1:x2]
    
    # Setup blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 5000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(mask_roi)
    
    detected_blobs = []
    for kp in keypoints:
        detected_blobs.append({
            'x': int(x1 + kp.pt[0]),
            'y': int(y1 + kp.pt[1]),
            'size': int(kp.size)
        })
    
    ball_count = len(keypoints)
    
    return ball_count, detected_blobs


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
def main():
    image_path = "images/map_test_2/3.jpg"
    
    # Step 1: Find ball clusters
    print("Step 1: Finding metal ball clusters...")
    frame, mask, contours = find_ball_clusters(image_path)
    
    if frame is None:
        return
    
    print(f"Found {len(contours)} clusters\n")
    
    # Create output image
    output = frame.copy()
    
    # Process each cluster
    for i, contour in enumerate(contours):
        print(f"=== Cluster {i+1} ===")
        
        # Step 2: Estimate distance
        distance_cm, pixel_diameter = estimate_distance(contour)
        print(f"Distance: {distance_cm:.1f} cm (pixel diameter: {pixel_diameter})")
        
        # Step 3: Count balls (choose one method)
        
        # Method A: YOLOv8 (requires trained model)
        ball_count, detections = count_balls_per_cluster_yolo(frame, contour, model_path='yolov5su.pt')
        
        # Method B: Hough Circle Transform (non-ML)
        # ball_count, circles = count_balls_per_cluster_hough(frame, mask, contour)
        
        # Method C: Blob Detection (alternative non-ML)
        # ball_count, blobs = count_balls_per_cluster_blob(frame, mask, contour)
        
        print(f"Ball count: {ball_count}")
        print()
        
        # Draw cluster bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw detected circles (if using Hough method)
        # for circle in circles:
        #     cv2.circle(output, (circle['x'], circle['y']), circle['radius'], (255, 0, 0), 2)
        #     cv2.circle(output, (circle['x'], circle['y']), 2, (0, 0, 255), 3)
        
        # Draw detected blobs (if using blob method)
        # for blob in blobs:
        #     cv2.circle(output, (blob['x'], blob['y']), blob['size']//2, (255, 0, 0), 2)
        #     cv2.circle(output, (blob['x'], blob['y']), 2, (0, 0, 255), 3)
        
        # Add labels
        label1 = f"Cluster {i+1}: {ball_count} balls"
        label2 = f"Distance: {distance_cm:.1f}cm"
        cv2.putText(output, label1, (x, y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output, label2, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
    
    # cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Detection Result', output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()