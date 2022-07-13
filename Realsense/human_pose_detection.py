"""
MIT License

Copyright (c) 2021 Florian Bruggisser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import time
from unittest import result

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
import pyrealsense2 as rs

OSC_ADDRESS = "/mediapipe/pose"
image_width = 640
image_height = 480
stream_fps = 30

def pixel_to_point(results, source, depth_frame, depth_intrin):
    test_x = results.pose_landmarks.landmark[source].x * image_width
    test_y = results.pose_landmarks.landmark[source].y * image_height

    depth = depth_frame.get_distance(int(test_x), int(test_y))
    result = rs.rs2_deproject_pixel_to_point(depth_intrin, [test_x, test_y], depth)
    return result

def send_pose(client: udp_client,
              landmark_list: landmark_pb2.NormalizedLandmarkList):
    if landmark_list is None:
        client.send_message(OSC_ADDRESS, 0)
        return

    # create message and send
    builder = OscMessageBuilder(address=OSC_ADDRESS)
    builder.add_arg(1)
    for landmark in landmark_list.landmark:
        builder.add_arg(landmark.x)
        builder.add_arg(landmark.y)
        builder.add_arg(landmark.z)
        builder.add_arg(landmark.visibility)
    msg = builder.build()
    client.send(msg)


def main():
    # read arguments
    parser = argparse.ArgumentParser()
    rs_group = parser.add_argument_group("RealSense")
    rs_group.add_argument("--resolution", default=[image_width, image_height], type=int, nargs=2, metavar=('width', 'height'),
                          help="Resolution of the realsense stream.")
    rs_group.add_argument("--fps", default=30, type=int,
                          help="Framerate of the realsense stream.")

    mp_group = parser.add_argument_group("MediaPipe")
    mp_group.add_argument("--model-complexity", default=1, type=int,
                          help="Set model complexity (0=Light, 1=Full, 2=Heavy).")
    mp_group.add_argument("--no-smooth-landmarks", action="store_false", help="Disable landmark smoothing.")
    mp_group.add_argument("--static-image-mode", action="store_true", help="Enables static image mode.")
    mp_group.add_argument("-mdc", "--min-detection-confidence", type=float, default=0.5,
                          help="Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful.")
    mp_group.add_argument("-mtc", "--min-tracking-confidence", type=float, default=0.5,
                          help=" Minimum confidence value ([0.0, 1.0]) to be considered tracked successfully.")

    nw_group = parser.add_argument_group("Network")
    nw_group.add_argument("--ip", default="127.0.0.1",
                          help="The ip of the OSC server")
    nw_group.add_argument("--port", type=int, default=7400,
                          help="The port the OSC server is listening on")

    args = parser.parse_args()

    # create osc client
    client = udp_client.SimpleUDPClient(args.ip, args.port)

    # setup camera loop
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        smooth_landmarks=args.no_smooth_landmarks,
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence)


    # create realsense pipeline
    pipeline = rs.pipeline()

    width, height = args.resolution

    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, stream_fps)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, stream_fps)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    prev_frame_time = 0

    # ====== Get depth Scale ======
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # ====== Set clipping distance ======
    clipping_distance_in_meters = 2
    clipping_distance = clipping_distance_in_meters / depth_scale




    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # ====== get conver data ======
            # same color_intrin == depth_intrin
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics


            if not color_frame:
                break
            if not depth_frame or not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # send the pose over osc
            send_pose(client, results.pose_landmarks)
  

            try:  
                source = mp_pose.PoseLandmark.NOSE
                result = pixel_to_point(results, source, depth_frame, depth_intrin)
                print(result)

            except:
                pass

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            cv2.putText(image, "FPS: %.0f" % fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('RealSense Pose Detector', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        pose.close()
        pipeline.stop()


if __name__ == "__main__":
    main()
