from controller import Supervisor, Keyboard
import numpy as np
import cv2

from perception.detector import CombinedBallDetector
from perception.ground_projector import GroundProjector
from perception.ball_tracker import BallTracker
from localization.april_tags import AprilTagDetector

# real-time spawning logic (stays in)
from webots_env.ball_spawner import spawn_balls
from webots_env.april_tags.april_tag_spawner import spawn_tags

# goal is to have no webots_env imports in the final simulation
# localization, aboslute position of the robot should be replaced with odometry logic
from webots_env.verify import find_nearest_ball_to_coord, robot_pose_from_supervisor
from webots_env.visualize import plot_nodes

TIME_STEP = 3

robot = Supervisor()

# keyboard initializationf
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Motor initialization
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

MAX_SPEED = 6.28

# Camera Initialization
camera = robot.getDevice('front_camera')
camera.enable(TIME_STEP)

width, height = camera.getWidth(), camera.getHeight()

# spawn balls
METAL_COUNT = 24
ORANGE_COUNT = 16
spawn_balls(robot, metal_count=METAL_COUNT, orange_count=ORANGE_COUNT, arena_size=2.0) # maybe softcode the arena size

# spawn april tags
spawn_tags(robot)

# ball detector
detector = CombinedBallDetector()
projector = GroundProjector(
    fov_h_rad=camera.getFov(),
    width=width,
    height=height,
    cam_height=0.16,
    pitch_down_rad=0.262,
)
orange_tracker = BallTracker(projector)
metal_tracker = BallTracker(projector)

# april tag detector
april_tag_detector = AprilTagDetector(fov=camera.getFov())


# simulation loop
while robot.step(TIME_STEP) != -1:

    key = keyboard.getKey()

    left = 0
    right = 0

    if key == Keyboard.UP:
        left = MAX_SPEED
        right = MAX_SPEED

    elif key == Keyboard.DOWN:
        left = -MAX_SPEED
        right = -MAX_SPEED

    elif key == Keyboard.LEFT:
        left = -MAX_SPEED
        right = MAX_SPEED

    elif key == Keyboard.RIGHT:
        left = MAX_SPEED
        right = -MAX_SPEED

    left_motor.setVelocity(left)
    right_motor.setVelocity(right)
    
    # Read camera after a single getkey
    image = camera.getImage()

    img = np.frombuffer(image, dtype=np.uint8)
    img = img.reshape((height, width, 4))

    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    frame = img_bgr # keep convention

    robot_x, robot_y, yaw = robot_pose_from_supervisor(robot)
    detections = detector.detect(frame)

    orange_balls = orange_tracker.update(detections['orange'], (robot_x, robot_y, yaw))
    # metal ball detection is worse because the balls are smaller and we have a smaller fov
    metal_balls = metal_tracker.update(detections['metal'], (robot_x, robot_y, yaw))

    ball_pos = [pos for _, pos in orange_balls + metal_balls if pos is not None]
    # plot_nodes(robot, ball_pos, points_are_world=True, orange_balls_count=len(orange_balls))

    data = april_tag_detector.detect(frame)
    print(data)

    cv2.imshow('Frame', frame)
    cv2.waitKey(1)


    
    # old visualization code (disregard)

    # store points for birds eye view visualization
    # projections = projector.locate_balls(bboxes)
    # points = [pos for _, pos in projections if pos is not None]
    # plot_nodes(robot, points)

    # for bbox, pos in projections:
    #     cv2.circle(frame, (bbox.cx, bbox.cy), bbox.radius, (0, 200, 255), 3)
    #     cv2.circle(frame, (bbox.cx, bbox.cy), 4,          (0,  80, 255), -1)

    #     if pos is not None:
    #         x, y = pos
    #         (real_x, real_y), error = find_nearest_ball_to_coord(robot, x, y)

    #         cv2.putText(frame, f"({x:.2f},{y:.2f}), e={error:.2f}", (bbox.x, bbox.y - 8),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            
    # cv2.putText(frame, f"Detected: {len(bboxes)}", (8, 28),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)
    
    # cv2.imshow("Camera with ball detection", frame)

    # cv2.waitKey(1)
