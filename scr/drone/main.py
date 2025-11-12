#!/usr/bin/env python3
import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from clover.srv import SetLEDEffect
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time
import os

# ---------------- ROS init ----------------
rospy.init_node('flight_with_segmentation')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

bridge = CvBridge()
current_frame = None

# ---------------- Подписка на камеру ----------------
def image_callback(msg):
    global current_frame
    current_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

rospy.Subscriber('/main_camera/image_raw', Image, image_callback)

# ---------------- Навигация ----------------
def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

# ---------------- Цветовая сегментация ----------------
def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Твои пороги цветов
    mask_r = cv2.inRange(hsv, (0,10,20), (35,255,255)) + cv2.inRange(hsv, (160,70,50), (179,255,255))
    mask_g = cv2.inRange(hsv, (55,120,80), (85,255,125))
    mask_y = cv2.inRange(hsv, (25,70,100), (35,255,255))
    # Морфологическая фильтрация (закрытие для удаления шумов)
    kernel = np.ones((5, 5), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, kernel)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)
    return mask_r, mask_g, mask_y

def segment_frame(frame, mask, color_bgr):
    color_layer = np.zeros_like(frame)
    color_layer[:] = color_bgr
    mask_3d = cv2.merge([mask, mask, mask])
    return np.where(mask_3d > 0, cv2.addWeighted(frame, 0.3, color_layer, 0.7, 0), frame)

# ---------------- Видео-запись ----------------
video_path = os.path.expanduser('~/segmentation_record.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

def process_and_record_frame(frame):
    global out
    if frame is None:
        return
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))

    mask_r, mask_g, mask_y = detect_colors(frame)
    frame = segment_frame(frame, mask_r, (0,0,255))
    frame = segment_frame(frame, mask_g, (0,255,0))
    frame = segment_frame(frame, mask_y, (0,255,255))

    # Центры масс
    for color_name, mask, color in [('', mask_r, (0,0,255)),
                                    ('', mask_g, (0,255,0)),
                                    ('', mask_y, (0,255,255))]:
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f'{color_name} ({cx},{cy})', (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out.write(frame)
    # Не вызываем imshow и waitKey — только запись

# ---------------- Поток обработки видео ----------------
def video_loop():
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if current_frame is not None:
            process_and_record_frame(current_frame)
        rate.sleep()

video_thread = threading.Thread(target=video_loop)
video_thread.start()

# ---------------- Полет ----------------
try:
    # Взлёт
    navigate_wait(z=1, frame_id='body', auto_arm=True)
    set_effect(r=0, g=0, b=0)

    navigate_wait(x=2, y=0, z=1.5, frame_id='aruco_map')

    # Желтый
    navigate_wait(x=1.5, y=0.5, z=1.5, frame_id='aruco_map')
    set_effect(r=255, g=255, b=0, effect='blink')
    rospy.sleep(3)
    set_effect(r=0, g=0, b=0)

    navigate_wait(x=0, y=0, z=2, frame_id='aruco_map')

    # Зеленый
    navigate_wait(x=0, y=1, z=1.5, frame_id='aruco_map')
    navigate_wait(x=0.5, y=1, z=1.5, frame_id='aruco_map')
    set_effect(r=0, g=255, b=0, effect='blink')
    rospy.sleep(3)
    set_effect(r=0, g=0, b=0)

    navigate_wait(x=2, y=1, z=1.5, frame_id='aruco_map')
    navigate_wait(x=2, y=2, z=1.5, frame_id='aruco_map')
    navigate_wait(x=0, y=2, z=1.5, frame_id='aruco_map')

    # Красный
    navigate_wait(x=0, y=3, z=1.5, frame_id='aruco_map')
    navigate_wait(x=1.2, y=3, z=1.5, frame_id='aruco_map')
    set_effect(r=255, g=0, b=0, effect='blink')
    rospy.sleep(3)
    set_effect(r=0, g=0, b=0)

    navigate_wait(x=2, y=3, z=1.5, frame_id='aruco_map')
    navigate_wait(x=2, y=3, z=1, frame_id='aruco_map')
    
    navigate_wait(x=2, y=0, z=1, frame_id='aruco_map')
    
    land()

finally:
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    rospy.signal_shutdown('Mission complete')

