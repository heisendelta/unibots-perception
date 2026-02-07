import cv2
import numpy as np

def detect_orange_balls():
    """
    Detect orange balls using webcam and draw bounding boxes around them.
    Uses HSV color space for robust color detection.
    """
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Orange Ball Detector Started")
    print("Press 'q' to quit")
    print("Press 's' to adjust sensitivity (opens trackbars)")
    
    # Flag for showing trackbars
    show_trackbars = False
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for orange color in HSV
        # Orange typically has Hue around 10-25 (in OpenCV's 0-179 range)
        # You may need to adjust these values based on lighting conditions
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
        
        # Filter and draw bounding boxes
        for contour in contours:
            # Filter by area to avoid small noise
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the shape is roughly circular (optional)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Only draw if shape is reasonably circular
                    if circularity > 0.5:
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Draw circle center
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        # Add label
                        label = f"Orange Ball ({int(area)})"
                        cv2.putText(frame, label, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Orange Ball Detection', frame)
        cv2.imshow('Mask', mask)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_trackbars = not show_trackbars
            if show_trackbars:
                print("Trackbar mode enabled - adjust HSV values in 'HSV Adjustment' window")
                create_trackbars()
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()

def create_trackbars():
    """
    Create trackbars for adjusting HSV threshold values.
    Useful for fine-tuning detection in different lighting conditions.
    """
    cv2.namedWindow('HSV Adjustment')
    
    # Create trackbars for lower HSV values
    cv2.createTrackbar('Lower H', 'HSV Adjustment', 5, 179, lambda x: None)
    cv2.createTrackbar('Lower S', 'HSV Adjustment', 100, 255, lambda x: None)
    cv2.createTrackbar('Lower V', 'HSV Adjustment', 100, 255, lambda x: None)
    
    # Create trackbars for upper HSV values
    cv2.createTrackbar('Upper H', 'HSV Adjustment', 25, 179, lambda x: None)
    cv2.createTrackbar('Upper S', 'HSV Adjustment', 255, 255, lambda x: None)
    cv2.createTrackbar('Upper V', 'HSV Adjustment', 255, 255, lambda x: None)
    
    print("Trackbars created! Adjust values to fine-tune detection.")

if __name__ == "__main__":
    detect_orange_balls()