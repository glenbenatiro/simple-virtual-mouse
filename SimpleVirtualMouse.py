"""
    SimpleVirtualMouse.py
    Copyright 2021 Louille Glen Benatiro

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import cv2
import time
import autopy
import numpy as np
import MediaPipeHands as mph

###

# definition of constants
cam_width, cam_height = 1280, 720
screen_width, screen_height = autopy.screen.size()
movement_bounding_box_margin = 200
smoothing_factor = 3
clicking_distance = 35

###

previous_time = 0
current_time = 0
previous_cursor_x, previous_cursor_y = 0, 0
current_cursor_x, current_cursor_y = 0, 0

###

capture = cv2.VideoCapture(0)
capture.set(3, cam_width)
capture.set(4, cam_height)

hand_detector = mph.MediaPipeHands(max_hands=1)

while True:
    # capture image from webcam
    success, img = capture.read()

    # flip the image horizontally (raw image is mirrored from webcam stream)
    img = cv2.flip(img, 1)

    # feed in image to hand detector
    img = hand_detector.detect_hands(img)

    # get coordinates of hand landmarks and the hand bounding box
    landmark_list, bounding_box = hand_detector.get_landmark_positions(img)

    # if hands are detected
    if len(landmark_list) != 0:
        # draw the bounding rectangle where mouse movement and click is permitted
        cv2.rectangle(img,
                      (movement_bounding_box_margin, movement_bounding_box_margin),
                      (cam_width - movement_bounding_box_margin, cam_height - movement_bounding_box_margin),
                      (0, 255, 0), 2)

        # get the coordinates of the tip of the index and middle finger
        index_x, index_y = landmark_list[8][1:3]
        x_middle, y_middle = landmark_list[12][1:3]

        # draw circle on tips of index and middle finger
        cv2.circle(img, landmark_list[8][1:3], 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, landmark_list[12][1:3], 10, (255, 0, 255), cv2.FILLED)

        # identify which fingers are currently up
        fingers = hand_detector.get_fingers_up()

        # if only index finger is up, moving mode
        if fingers == [0, 1, 0, 0, 0]:
            # convert the index finger tip location coordinates to mouse pointer coordinates on the screen
            current_cursor_x = np.interp(index_x, (movement_bounding_box_margin, cam_width - movement_bounding_box_margin), (0, screen_width))
            current_cursor_y = np.interp(index_y, (movement_bounding_box_margin, cam_height - movement_bounding_box_margin), (0, screen_height))

            # do smoothing on mouse pointer coordinates
            if smoothing_factor > 0:
                current_cursor_x = previous_cursor_x + (current_cursor_x - previous_cursor_x) / smoothing_factor
                current_cursor_y = previous_cursor_y + (current_cursor_y - previous_cursor_y) / smoothing_factor

            # move mouse pointer given coordinates
            autopy.mouse.move(current_cursor_x, current_cursor_y)
            previous_cursor_x, previous_cursor_y = current_cursor_x, current_cursor_y

        # if both index and middle finger is up, check if both are in contact with each other = mouse click
        elif fingers == [0, 1, 1, 0, 0]:
            # find the distance between the index finger tip and middle finger tip
            distance, img, distance_info = hand_detector.get_distance(8, 12, img)
            print("Index and Middle Distance: " + str(distance))

            # check if both are in contact
            if distance < clicking_distance:
                cv2.circle(img, (distance_info[4], distance_info[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # calculate and display fps
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # display image
    cv2.imshow("output", img)
    cv2.waitKey(1)
