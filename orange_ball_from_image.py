import cv2
import numpy as np

# Read image from file
frame = cv2.imread("images/five_orange_balls.jpg")

if frame is None:
    print("Error: Could not read image")
    exit()

# Camera/ball parameters
KNOWN_BALL_DIAMETER_CM = 4.0  # Ping pong ball diameter is 40mm = 4cm
FOCAL_LENGTH_PIXELS = 800  # Calibrate this based on your camera

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

# Size threshold
MIN_AREA = 5000  # Constant threshold for now

# Calculate reference area for a single ping pong ball at 10cm
# This will be calibrated from the smallest detected cluster
reference_areas = []

print("\n=== Orange Ball Detection Results ===")
cluster_count = 0

# First pass: collect areas to estimate single ball area
temp_areas = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > MIN_AREA:
        temp_areas.append(area)

# Estimate single ball area as the minimum area detected (likely a single ball)
if temp_areas:
    SINGLE_BALL_AREA = min(temp_areas)
else:
    SINGLE_BALL_AREA = MIN_AREA

print(f"Reference single ball area: {SINGLE_BALL_AREA:.0f} pixels")

# Process each cluster
for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > MIN_AREA:
        cluster_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        
        # 1. Cluster area
        cluster_area = area
        
        # 2. Estimate number of balls per cluster
        # More realistic estimation based on actual ping pong ball area
        estimated_balls = max(1, round(area / SINGLE_BALL_AREA))
        
        # Additional heuristic: if cluster is very elongated, likely multiple balls
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.0 and estimated_balls == 1:
            estimated_balls = 2
        
        # 3. Estimate distance from camera
        pixel_diameter = max(w, h)
        
        if pixel_diameter > 0:
            distance_cm = (KNOWN_BALL_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / pixel_diameter
        else:
            distance_cm = 0
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add detailed label
        ball_text = "ball" if estimated_balls == 1 else "balls"
        label1 = f"Cluster {cluster_count}: {estimated_balls} {ball_text}"
        label2 = f"Dist: {distance_cm:.1f}cm"
        
        cv2.putText(frame, label1, (x, y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, label2, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print to console
        print(f"\nCluster {cluster_count}:")
        print(f"  Area: {cluster_area:.0f} pixels ({cluster_area/SINGLE_BALL_AREA:.1f}x reference)")
        print(f"  Estimated balls: {estimated_balls}")
        print(f"  Bounding box: {w}x{h} pixels (aspect ratio: {aspect_ratio:.2f})")
        print(f"  Estimated distance: {distance_cm:.1f} cm")

print(f"\nTotal clusters detected: {cluster_count}")
print("=" * 40)

# Make windows resizable
cv2.namedWindow('Orange Ball Detection', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

# Display the result
cv2.imshow('Orange Ball Detection', frame)
# cv2.imshow('Mask', mask)

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()