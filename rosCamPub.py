#!/usr/bin/env python
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from lane import *
import imutils
import cv2
from collections import deque

from hyperloop.msg import camout

def publish_message():
    pub = rospy.Publisher('camout', camout, queue_size=10)
    speed = 10 # operating speed in % PWM
    
    #Variables to be updated each loop
    lastTime = 0 
    lastError = 0
    
    # PD constants
    Kp = 0.4
    Kd = Kp * 0.65
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('video_pub_py', anonymous=True)
    # Go through the loop 10 times per second
    rate = rospy.Rate(10) # 10hz
    # Create a VideoCapture object
    # The argument '0' gets the default webcam.
    # Used to convert between ROS and OpenCV images
    
    print("[INFO] sampling THREADED frames from `picamera` module...")
    vs = PiVideoStream().start()
    q=deque([0,0,0,0,0])

    # PD constants
    Kp = 0.4
    Kd = Kp * 0.65

    # While ROS is still running.
    while not rospy.is_shutdown():
    
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        frame = vs.read()
        frame = imutils.resize(frame, width=640, height=480)
        avg=0
        temp=camout()
        if frame is not None:
            frame,line_segments = find_and_draw_lanes(frame)
            lane_lines = average_slope_intercept(frame,line_segments)
            lane_lines_image = display_lines(frame,lane_lines)
            steering_angle = get_steering_angle(frame, lane_lines)
            heading_image = display_heading_line(lane_lines_image,steering_angle)
            
            now = time.time() # current time variable
            dt = now - lastTime
            deviation = steering_angle - 90 # equivalent to angle_to_mid_deg variable
            error = abs(deviation)

            derivative = kd * (error - lastError) / dt 
            proportional = kp * error
            PD = int(speed + derivative + proportional)

            spd = abs(PD)
            if spd > 25:
                spd = 25
            lastError = error
            q.append(steering_angle)
            dummy=q.popleft()
            avg=int(sum(q)/5)
            speed=10
            print(avg," avg ")
            if (avg<-45):
                avg=-45
            if (avg>35):
                avg=35
        else:
            speed = 0
            avg = 0

        temp.speed = speed
        temp.direction = avg
        pub.publish(temp)
            
        # Sleep just enough to maintain the desired rate
        rate.sleep()

if __name__ == '__main__':
  try:
    publish_message()
  except rospy.ROSInterruptException:
    pass
