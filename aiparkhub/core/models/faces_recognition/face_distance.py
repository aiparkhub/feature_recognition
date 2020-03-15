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
# @Program : 人像面部识别 - 面距 | Face Recognition-Face Spacing
# @File : face_distance.py
# @Description : 人像面部识别 - 面距 | Face Recognition-Face Spacing
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.


# 导入 自定义模块 | Import Custom Module
import face_recognition

'''
Define training dataset
'''
ABSOLUTE_PATH = 'Your_native_absolute_path'  # Should set absolute path
COMMON_PATH = ABSOLUTE_PATH + 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_1 = COMMON_PATH + 'president_obama/dataset_for_photo/obama.jpg'
RESOURCE_PATH_2 = COMMON_PATH + 'president_obama/dataset_for_photo/biden.jpg'
RESOURCE_PATH_3 = COMMON_PATH + 'president_obama/dataset_for_photo/two_people.jpg'


"""
Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
You can do that by using the face_distance function.

The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
positive matches at the risk of more false negatives.

Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
smaller distance are more similar to each other than ones with a larger distance.
"""


def face_distance(example_resource_1,
                  example_resource_2,
                  example_resource_3):

    # Load some images to compare against
    known_obama_image = face_recognition.load_image_file(example_resource_1)
    known_biden_image = face_recognition.load_image_file(example_resource_2)

    # Get the face encodings for the known images
    obama_face_encoding = face_recognition.face_encodings(known_obama_image)[0]
    biden_face_encoding = face_recognition.face_encodings(known_biden_image)[0]

    known_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]

    # Load a test image and get encondings for it
    image_to_test = face_recognition.load_image_file(example_resource_3)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(
        known_encodings, image_to_test_encoding)

    for i, face_distance in enumerate(face_distances):
        print(
            "The test image has a distance of {:.2} from known image #{}".format(
                face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(
            face_distance < 0.5))
        print()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 函数 | Call function
    face_distance(
        RESOURCE_PATH_1,
        RESOURCE_PATH_2,
        RESOURCE_PATH_3)
