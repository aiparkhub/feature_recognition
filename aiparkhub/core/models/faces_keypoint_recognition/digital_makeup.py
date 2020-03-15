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
# @Program : 人脸关键点识别 - 涂美妆  | Face key recognition-applying makeup
# @File : digital_makeup.py
# @Description : 人脸关键点识别 - 为人像面部涂美妆 | Face Key Recognition-Applying Beauty to Portrait Faces
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.

# 导入 第三方模块 | Import third-party modules
from PIL import Image, ImageDraw

# 导入 自定义模块 | Import Custom Module
import face_recognition


'''
Define training dataset
'''
ABSOLUTE_PATH = 'Your_native_absolute_path/'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'president_obama/dataset_for_photo/biden.jpg'
RESOURCE_PATH_1 = COMMON_PATH + 'momoland/dataset_for_photo/momoland_nancy_v3.png'
RESOURCE_PATH_2 = COMMON_PATH + 'president_obama/dataset_for_photo/two_people.jpg'


def digital_makeup(example_photo):
    """
    定义 人脸关键点识别 函数 | Define face keypoint recognition function
    :param example_photo:
    :return:
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(example_photo)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'],
               fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] +
               [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] +
               [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        pil_image.show()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 digital_makeup 函数 | Call digital_makeup function
    digital_makeup(RESOURCE_PATH_2)
