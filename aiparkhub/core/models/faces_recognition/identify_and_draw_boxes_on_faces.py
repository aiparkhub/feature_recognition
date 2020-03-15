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
# @Program : 人像面部识别 - 绘制标示框并标注姓名 | Face Recognition-Draw Box and Name
# @File : identify_and_draw_boxes_on_faces.py
# @Description : 人像面部识别 - 绘制标示框并标注姓名 | Face Recognition-Draw Box and Name
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.

# 导入 第三方模块 | Import third-party modules
from PIL import Image, ImageDraw
import numpy as np

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


def identify_and_draw_boxes_on_faces(
        example_resource_1,
        example_resource_2,
        example_resource_3):
    """
    This is an example of running face recognition on a single image and drawing a box around each person that was identified.
    :return:
    """
    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file(example_resource_1)
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(example_resource_2)
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "Barack Obama",
        "Joe Biden"
    ]

    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(example_resource_3)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(
        unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new
        # face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10),
                        (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5),
                  name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

    # You can also save a copy of the new image to disk if you want by uncommenting this line
    # pil_image.save("image_with_boxes.jpg")


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # 调用 函数 | Call function
    identify_and_draw_boxes_on_faces(
        RESOURCE_PATH_1,
        RESOURCE_PATH_2,
        RESOURCE_PATH_3)
