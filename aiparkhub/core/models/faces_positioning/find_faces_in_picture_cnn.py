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
# @Program : 人像面部定位 - 卷积神经网络深度学习模型 定位人像面部  | Portrait Face Localization-Portrait Face Localization-Convolutional Neural Network Deep Learning Model Locate Portrait Face
# @File : find_faces_in_picture_cnn.py
# @Description : 人像面部定位 - 卷积神经网络深度学习模型 定位人像面部  | Portrait Face Localization-Portrait Face Localization-Convolutional Neural Network Deep Learning Model Locate Portrait Face
# @Copyright © 2019 - 2020 AParkHub-Organization. All rights reserved.


# 导入 第三方模块 | Import third-party modules
from PIL import Image

# 导入 自定义模块 | Import Custom Module
import face_recognition

'''
Define training dataset
'''
ABSOLUTE_PATH = '/Users/system/home/work/develop/code_flow/work_projects/git_flow/aiparkhub/feature_recognition/'
# ABSOLUTE_PATH = 'Your_native_absolute_path'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'momoland/dataset_for_photo/momoland_daisy.jpg'
RESOURCE_PATH_1 = COMMON_PATH + 'momoland/dataset_for_photo/momoland_daisy_v2.jpg'
RESOURCE_PATH_2 = COMMON_PATH + 'president_obama/dataset_for_photo/two_people.jpg'


def find_faces_in_picture_cnn(example_photo):
    """
    定义 卷积神经网络深度学习模型 定位人像面部 函数 | Definition Convolutional Neural Network Deep Learning Model Locate Portrait Face Function
    :param example_photo:
    :return:
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(example_photo)

    """
    Find all the faces in the image using a pre-trained convolutional neural network.
    This method is more accurate than the default HOG model, but it's slower
    unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do, this will use GPU acceleration and perform well.
    See also: find_faces_in_picture.py
    """
    face_locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print(
            "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                top,
                left,
                bottom,
                right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 函数 | Call function
    find_faces_in_picture_cnn(RESOURCE_PATH_1)
