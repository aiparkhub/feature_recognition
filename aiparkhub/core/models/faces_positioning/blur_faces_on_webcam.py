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
# @Program : 人像面部定位 - 网络摄像头视频 添加 面部高斯模糊
# @File : blur_faces_on_webcam.py
# @Description : Portrait Facial Location-Webcam Video Add Facial Gaussian Blur
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.

# 导入 第三方模块 | Import third-party modules
import cv2
import emoji as em

# 导入 自定义模块 | Import Custom Module
import face_recognition

# Draw font style
font = cv2.FONT_ITALIC

'''
Define training dataset
'''
ABSOLUTE_PATH = 'Your_native_absolute_path'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_1 = COMMON_PATH + 'momoland/dataset_for_video/momoland_fancam_baam.flv'
RESOURCE_PATH_2 = 0

"""
This is a demo of blurring faces in video;

PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
"""


def blur_faces_on_webcam(example_resource):
    """
    定义 网络摄像头视频 添加 面部高斯模糊 函数 | Define webcam video add face gaussian blur function
    :param example_resource:
    :return:
    """

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(example_resource)

    # Initialize some variables
    face_locations = []

    while True:
        # Print log information
        print(
            em.emojize(
                '================================== Deep Learning - AiParkHub Organization (Feature Recognition)=> :satellite:  Training <Feature Model> :bar_chart:  ==========================',
                use_aliases=True))
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Settings window full screen
        cv2.namedWindow(
            'Deep Learning - AiParkHub Organization (Feature Recognition)',
            cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            'Deep Learning - AiParkHub Organization (Feature Recognition)',
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)

        # Settings local window
        # cv2.namedWindow("Deep Learning - AiParkHub Organization (Feature Recognition)", 0)
        # cv2.resizeWindow("Deep Learning - AiParkHub Organization (Feature Recognition)", 1920, 1080)
        # cv2.moveWindow("Deep Learning - AiParkHub Organization (Feature Recognition)", 350, 150)

        # Resize frame of video to 1/4 size for faster face detection
        # processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(
            small_frame, model="cnn")

        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was
            # scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Extract the region of the image that contains the face
            face_image = frame[top:bottom, left:right]

            # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

            print(
                em.emojize(
                    'Feature Recognition info => :chart_with_upwards_trend:',
                    use_aliases=True),
                f' DataSet[{top}, {right}, {bottom}, {left}] - Feature Counts => ' +
                str(
                    len(face_locations)))

            '''
            Visual information Style - 1 For 1080P Media
            '''
            cv2.putText(
                frame,
                "Deep Learning - AiParkHub Organization (Feature Recognition)",
                (20,
                 40),
                font,
                1.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA)
            cv2.putText(frame, "Q: Quit", (20, 750), font,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Put the blurred face region back into the frame image
            frame[top:bottom, left:right] = face_image

        # Display the resulting image
        cv2.imshow(
            'Deep Learning - AiParkHub Organization (Feature Recognition)',
            frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(
                em.emojize(
                    '================================== Deep Learning - AiParkHub Organization (Feature Recognition) => :tada:  All Done! :tada:  ==========================',
                    use_aliases=True))
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 函数 | Call function
    blur_faces_on_webcam(RESOURCE_PATH_1)
