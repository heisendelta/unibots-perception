import cv2
import numpy as np

# ============================================================================
# FUNCTION 1: Find ball clusters (mask of the balls)
# ============================================================================
def find_ball_clusters(image_path):
    """
    Find orange ball clusters using HSV color segmentation.
    
    Args:
        image_path: Path to input image
        
    Returns:
        frame: Original image
        mask: Binary mask of orange regions
        contours: List of contours for each cluster
    """
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Could not read image")
        return None, None, None
    
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Create mask for orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    MIN_AREA = 500
    filtered_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    
    return frame, mask, filtered_contours


# ============================================================================
# FUNCTION 2: Estimate distance from camera to cluster (depth estimation)
# ============================================================================
def estimate_distance(contour, focal_length=800, real_ball_diameter_cm=4.0):
    """
    Estimate distance to a ball cluster using pinhole camera model.
    
    Args:
        contour: OpenCV contour of the cluster
        focal_length: Camera focal length in pixels (needs calibration)
        real_ball_diameter_cm: Real diameter of ping pong ball in cm
        
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
    - A trained YOLOv8 model for ping pong balls
    
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
    
    Args:
        frame: Original image
        mask: Binary mask of orange regions
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
    
    # Apply mask to focus only on orange regions
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_roi)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # Minimum distance between circle centers
        param1=50,   # Canny edge threshold
        param2=30,   # Accumulator threshold (lower = more circles detected)
        minRadius=10,
        maxRadius=100
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


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
def main():
    image_path = "images/map_test_1/image5.jpg"
    
    # Step 1: Find ball clusters
    print("Step 1: Finding ball clusters...")
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
        
        print(f"Ball count: {ball_count}")
        print()
        
        # Draw cluster bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw detected circles (if using Hough method)
        # for circle in circles:
        #     cv2.circle(output, (circle['x'], circle['y']), circle['radius'], (255, 0, 0), 2)
        #     cv2.circle(output, (circle['x'], circle['y']), 2, (0, 0, 255), 3)
        
        # Add labels
        label1 = f"Cluster {i+1}: {ball_count} balls"
        label2 = f"Distance: {distance_cm:.1f}cm"
        cv2.putText(output, label1, (x, y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output, label2, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
    
    # cv2.imshow('Original', frame)
    # cv2.imshow('Mask', mask)
    cv2.imshow('Detection Result', output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()