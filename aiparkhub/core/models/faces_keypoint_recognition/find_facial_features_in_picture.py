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
# @Program : 人脸关键点识别 - 提取人像面部关键点 | Face keypoint recognition-extract keypoints from portrait faces
# @File : find_facial_features_in_picture.py
# @Description : 人脸关键点识别 - 提取人像面部关键点 | Face keypoint recognition-extract keypoints from portrait faces
# @Copyright © 2019 - 2020 AParkHub-Organization. All rights reserved.

# 导入 第三方模块 | Import third-party modules
from PIL import Image, ImageDraw

# 导入 自定义模块 | Import Custom Module
import face_recognition


'''
Define training dataset
'''
ABSOLUTE_PATH = 'Your_native_absolute_path/'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_1 = COMMON_PATH + 'president_obama/dataset_for_photo/two_people.jpg'
RESOURCE_PATH_2 = COMMON_PATH + 'momoland/dataset_for_photo/momoland_nayun.jpg'
RESOURCE_PATH_3 = COMMON_PATH + 'momoland/dataset_for_photo/momoland_yeonwoo.jpg'


def extract_key_points_portrait_faces(example_photo):
    """
    定义 提取人像面部关键点 函数 | Definition function for extracting key points from portrait faces
    :param example_photo:
    :return:
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(example_photo)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    print(
        "I found {} face(s) in this photograph.".format(
            len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(
                facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)

    # Show the picture
    pil_image.show()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 extract_key_points_portrait_faces 函数 | Call extract_key_points_portrait_faces function
    extract_key_points_portrait_faces(RESOURCE_PATH_3)
