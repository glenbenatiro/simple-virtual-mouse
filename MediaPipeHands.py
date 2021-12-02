"""
    MediaPipeHands.py
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
import mediapipe as mp
import time
import math


class MediaPipeHands:
    def __init__(self,
                 mode=False,
                 max_hands=2,
                 model_complexity=1,
                 detection_confidence=0.7,
                 tracking_confidence=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # initiate actual MediaPipe Hands object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.max_hands,
                                        self.model_complexity,
                                        self.detection_confidence,
                                        self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        # define the index of the tips of the 5 fingers
        self.fingertip_IDs = [4, 8, 12, 16, 20]

        # declare the variables to be used in the class
        self.bounding_box_margin = 20
        self.landmark_list = []
        self.results = []

    def detect_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def get_landmark_positions(self, img, hand_number=0, draw=False):
        x_list = []
        y_list = []
        bounding_box = []
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_number]

            for i, landmarks in enumerate(selected_hand.landmark):
                img_height, img_width, img_channels = img.shape
                current_x, current_y = int(landmarks.x * img_width), int(landmarks.y * img_height)

                x_list.append(current_x)
                y_list.append(current_y)

                for idx, hand_handedness in enumerate(self.results.multi_handedness):
                    if hand_handedness.classification[0].label == "Left":
                        handedness = 0
                    else:
                        handedness = 1

                    self.landmark_list.append([i, current_x, current_y, handedness])

                if draw:
                    cv2.circle(img, (current_x, current_y), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min, y_min, x_max, y_max
            cv2.rectangle(img,
                          (bounding_box[0] - self.bounding_box_margin, bounding_box[1] - self.bounding_box_margin),
                          (bounding_box[2] + self.bounding_box_margin, bounding_box[3] + self.bounding_box_margin),
                          (255, 255, 0),
                          2)

        return self.landmark_list, bounding_box

    def get_fingers_up(self):
        fingers = []

        # when using left hand
        if self.landmark_list[0][3] == 0:
            if self.landmark_list[self.fingertip_IDs[0]][1] > self.landmark_list[self.fingertip_IDs[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # when using right hand
        else:
            if self.landmark_list[self.fingertip_IDs[0]][1] < self.landmark_list[self.fingertip_IDs[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 other fingers
        for id in range(1, 5):
            if self.landmark_list[self.fingertip_IDs[id]][2] < self.landmark_list[self.fingertip_IDs[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        distance = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return distance, img, [x1, y1, x2, y2, cx, cy]
