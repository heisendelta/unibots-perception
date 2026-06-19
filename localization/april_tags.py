import math

import cv2
import numpy as np
from pupil_apriltags import Detector

class AprilTagDetector:
    def __init__(self, fov=0.785):
        self.detector = Detector(families="tag36h11", quad_decimate=1.0)  # 1.0, not 2.0, so small/far tags still decode
        self.fov = fov

    def detect(self, frame: np.ndarray) -> list[int]:
        H, W = frame.shape[:2]

        fx = (W / 2) / math.tan(self.fov / 2)
        fy = fx
        cx = W / 2
        cy = H / 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detect(gray, estimate_tag_pose=True,
                            camera_params=(fx, fy, cx, cy), tag_size=0.1)  # 100 mm
        
        data = []

        for d in dets:
            tag_id = d.tag_id      # 0..23
            R = d.pose_R           # 3x3, tag orientation in the camera frame
            t = d.pose_t

            data.append(tag_id, R, t)

        return data  

