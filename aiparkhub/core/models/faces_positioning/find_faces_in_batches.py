# -*- coding:utf-8 -*-
#
# Geek International Park | 极客国际公园 & GeekParkHub | 极客实验室
# Website | https://www.geekparkhub.com
# Description | Open · Creation |
# Open Source Open Achievement Dream, GeekParkHub Co-construction has never been seen before.
#
# HackerParkHub | 黑客公园
# Website | https://www.hackerparkhub.org
# Description | In the spirit of fearless exploration, create unknown technology and worship of technology.
#
# AIParkHub | 人工智能公园
# Website | https://github.com/aiparkhub
# Description | Embark on the wave of AI and push the limits of machine intelligence.
#
# @GeekDeveloper : JEEP-711
# @Author : system
# @Version : 0.2.5
# @Program : 人像面部定位 - 卷积神经网络深度学习模型 批量识别 资源中的人像面部
# @File : find_faces_in_batches.py
# @Description : Portrait Face Localization-Convolutional Neural Network Deep Learning Model Batch Recognize Portrait Faces in Photos
# @Copyright © 2019 - 2020 AParkHub-Organization. All rights reserved.


# 导入 第三方模块 | Import third-party modules
import cv2

# 导入 自定义模块 | Import Custom Module
import face_recognition

'''
Define training dataset
'''
ABSOLUTE_PATH = '/Users/system/home/work/develop/code_flow/work_projects/git_flow/aiparkhub/feature_recognition/'
# ABSOLUTE_PATH = 'Your_native_absolute_path'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'momoland/dataset_for_video/momoland_fancam_baam.flv'


"""
This core finds all faces in a list of images using the CNN model.
This demo is for the _special case_ when you need to find faces in LOTS of images very quickly and all the images
are the exact same size. This is common in video processing applications where you have lots of video frames to process.

If you are processing a lot of images and using a GPU with CUDA, batch processing can be ~3x faster then processing
single images at a time. But if you aren't using a GPU, then batch processing isn't going to be very helpful.

PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read the video file.
OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
"""


def find_faces_in_batches(example_resource):
    """
    定义 卷积神经网络深度学习模型 批量 识别资源中的人像面部 函数
    Definition Convolutional neural network deep learning model Batch recognition of portrait face functions in resources
    """

    # Open video file
    video_capture = cv2.VideoCapture(example_resource)
    frames = []
    frame_count = 0

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        # (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        # Every 128 frames (the default batch size), batch process the list of
        # frames to find faces
        if len(frames) == 128:
            batch_of_face_locations = face_recognition.batch_face_locations(
                frames, number_of_times_to_upsample=0)

            # Now let's list all the faces we found in all 128 frames
            for frame_number_in_batch, face_locations in enumerate(
                    batch_of_face_locations):
                number_of_faces_in_frame = len(face_locations)

                frame_number = frame_count - 128 + frame_number_in_batch
                print(
                    "I found {} face(s) in frame #{}.".format(
                        number_of_faces_in_frame,
                        frame_number))

                for face_location in face_locations:
                    # Print the location of each face in this frame
                    top, right, bottom, left = face_location
                    print(
                        " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                                                                                                                 bottom,
                                                                                                                 right))

            # Clear the frames array to start the next batch
            frames = []


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 函数 | Call function
    find_faces_in_batches(RESOURCE_PATH_0)
