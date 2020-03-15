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
# @Program : 人脸识别 | Face recognition
# @File : feature_recognition.py
# @Description : 获取资源数据集进行人脸识别 | Obtaining a resource dataset for face recognition
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.


# 导入 第三方模块 | Import third-party modules
import cv2
import numpy as np
import emoji as em

# 导入 自定义模块 | Import Custom Module
import face_recognition

# Draw font style
font = cv2.FONT_ITALIC

'''
Define training dataset
'''
COMMON_PATH = 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'momoland/dataset_for_video/momoland_fancam_baam.flv'


def fast_training_model(training_dataset):
    """
    定义 (快速版本) 训练模型 函数 | Define (fast version) training model functions
    """

    # Get a reference to webcam #0 (the default one)
    resource_capture = cv2.VideoCapture(training_dataset)

    # Definition resource width and height
    w = int(resource_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + 1
    h = int(resource_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 1

    # Recognize portrait faces from resources and output recognition results as new video files
    resourcesWriter = cv2.VideoWriter(
        'aiparkhub/core/data/feature_training_dataset/part_1/momoland/rendering_data/rendering_test_training_part_1.mp4',
        cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60, (w, h))

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file(
        COMMON_PATH + "president_obama/dataset_for_photo/obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(
        COMMON_PATH + "president_obama/dataset_for_photo/biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # ==============================================================================================================
    # momoland_jooe_photo
    momoland_jooe_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_jooe.jpg")
    jooe_face_encoding = face_recognition.face_encodings(momoland_jooe_photo)[
        0]

    # momoland_hyebin_photo
    momoland_hyebin_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_hyebin_v2.jpg")
    hyebin_face_encoding = face_recognition.face_encodings(momoland_hyebin_photo)[
        0]

    # momoland_jane_photo
    momoland_jane_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_jane.jpg")
    jane_face_encoding = face_recognition.face_encodings(momoland_jane_photo)[
        0]

    # momoland_nayun_photo
    momoland_nayun_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_nayun.jpg")
    nayun_face_encoding = face_recognition.face_encodings(momoland_nayun_photo)[
        0]

    # momoland_ahin_photo
    momoland_ahin_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_ahin_v2.jpg")
    ahin_face_encoding = face_recognition.face_encodings(momoland_ahin_photo)[
        0]

    # momoland_nancy_photo
    momoland_nancy_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_nancy_v3.png")
    nancy_face_encoding = face_recognition.face_encodings(momoland_nancy_photo)[
        0]

    # momoland_taeha_photo
    momoland_taeha_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_taeha_v2.png")
    taeha_face_encoding = face_recognition.face_encodings(momoland_taeha_photo)[
        0]

    # momoland_daisy_photo
    momoland_daisy_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_daisy_v2.jpg")
    daisy_face_encoding = face_recognition.face_encodings(momoland_daisy_photo)[
        0]

    # momoland_yeonwoo_photo
    momoland_yeonwoo_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_yeonwoo_v2.jpg")
    yeonwoo_face_encoding = face_recognition.face_encodings(
        momoland_yeonwoo_photo)[0]
    # ==============================================================================================================

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding,
        jooe_face_encoding,
        hyebin_face_encoding,
        jane_face_encoding,
        nayun_face_encoding,
        ahin_face_encoding,
        nancy_face_encoding,
        taeha_face_encoding,
        daisy_face_encoding,
        yeonwoo_face_encoding
    ]

    known_face_names = [
        "Barack Obama",
        "Joe Biden",
        "Joo E",
        "Hye Bin",
        "Jane",
        "Na Yun",
        "Ah In",
        "Nancy",
        "Taeha",
        "Daisy",
        "Yeon Woo"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    try:
        # Definition capture screen
        while True:
            # Print log information
            print(
                em.emojize(
                    '================================== Deep Learning - AiParkHub Organization (Feature Recognition)=> :satellite:  Training <Feature Model> :bar_chart:  ==========================',
                    use_aliases=True))

            # Grab a single frame of video
            ret, frame = resource_capture.read()

            # Settings window full screen
            cv2.namedWindow(
                'Deep Learning - AiParkHub Organization (Feature Recognition)',
                cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'Deep Learning - AiParkHub Organization (Feature Recognition)',
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

            # Resize frame of video to 1/4 size for faster face recognition
            # processing, interpolation=cv2.INTER_NEAREST
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of
                # video
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance
                    # to the new face
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (
                    top, right, bottom, left), name in zip(
                    face_locations, face_names):
                # Scale back up face locations since the frame we detected in
                # was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(
                    frame, (left, top), (right, bottom), (255, 0, 0), 2)

                # Portrait Face Statistics
                cv2.putText(frame, "Feature Counts : " + str(len(face_locations)),
                            (20, 100), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

                cv2.putText(frame, "Stage Name : " + str(name),
                            (20, 150), font, 1, (0, 255, 0), 3, cv2.LINE_AA)

                print(
                    em.emojize(
                        'Feature Recognition info => :chart_with_upwards_trend:',
                        use_aliases=True),
                    f' DataSet[{top}, {right}, {bottom}, {left}] - Feature Counts => ' +
                    str(
                        len(face_locations)),
                    f', Stage Name[{name}]')

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            font, 1.0, (255, 255, 255), 2)

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
                (255,
                 255,
                 255),
                2,
                cv2.LINE_AA)
            cv2.putText(frame, "Q: Quit Training Model", (20, 750),
                        font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # '''
            # Visual information Style - 2 For 4K
            # '''
            # cv2.putText(frame, "Deep Learning - AiParkHub Organization (Feature Recognition)", (20, 40), font, 1.5,
            #             (240, 237, 232), 3, cv2.LINE_AA)
            # cv2.putText(frame, "Q: Quit Training Model", (20, 950), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Perform resource write
            resourcesWriter.write(frame)

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
    except Exception as e:
        print('Exception => ', e)
    finally:
        # Release resources
        resource_capture.release()
        cv2.destroyAllWindows()


def slow_training_model(training_dataset):
    """
    定义 (慢速版本)  训练模型 函数 | Define (slow version) training model function
    """

    # Get a reference to webcam #0 (the default one)
    resource_capture = cv2.VideoCapture(training_dataset)

    # Definition resource width and height
    w = int(resource_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + 1
    h = int(resource_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 1

    # Recognize portrait faces from resources and output recognition results as new video files
    resourcesWriter = cv2.VideoWriter(
        'aiparkhub/core/data/feature_training_dataset/part_1/momoland/rendering_data/rendering_test_training_part_2.mp4',
        cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60, (w, h))

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file(
        COMMON_PATH + "president_obama/dataset_for_photo/obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(
        COMMON_PATH + "joe_biden/dataset_for_photo/biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # ==============================================================================================================
    # momoland_jooe_photo
    momoland_jooe_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_jooe.jpg")
    jooe_face_encoding = face_recognition.face_encodings(momoland_jooe_photo)[
        0]

    # momoland_hyebin_photo
    momoland_hyebin_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_hyebin_v2.jpg")
    hyebin_face_encoding = face_recognition.face_encodings(momoland_hyebin_photo)[
        0]

    # momoland_jane_photo
    momoland_jane_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_jane.jpg")
    jane_face_encoding = face_recognition.face_encodings(momoland_jane_photo)[
        0]

    # momoland_nayun_photo
    momoland_nayun_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_nayun.jpg")
    nayun_face_encoding = face_recognition.face_encodings(momoland_nayun_photo)[
        0]

    # momoland_ahin_photo
    momoland_ahin_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_ahin_v2.jpg")
    ahin_face_encoding = face_recognition.face_encodings(momoland_ahin_photo)[
        0]

    # momoland_nancy_photo
    momoland_nancy_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_nancy_v3.png")
    nancy_face_encoding = face_recognition.face_encodings(momoland_nancy_photo)[
        0]

    # momoland_taeha_photo
    momoland_taeha_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_taeha_v2.png")
    taeha_face_encoding = face_recognition.face_encodings(momoland_taeha_photo)[
        0]

    # momoland_daisy_photo
    momoland_daisy_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_daisy_v2.jpg")
    daisy_face_encoding = face_recognition.face_encodings(momoland_daisy_photo)[
        0]

    # momoland_yeonwoo_photo
    momoland_yeonwoo_photo = face_recognition.load_image_file(
        COMMON_PATH + "momoland/dataset_for_photo/momoland_yeonwoo_v2.jpg")
    yeonwoo_face_encoding = face_recognition.face_encodings(
        momoland_yeonwoo_photo)[0]
    # ==============================================================================================================

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding,
        jooe_face_encoding,
        hyebin_face_encoding,
        jane_face_encoding,
        nayun_face_encoding,
        ahin_face_encoding,
        nancy_face_encoding,
        taeha_face_encoding,
        daisy_face_encoding,
        yeonwoo_face_encoding
    ]

    known_face_names = [
        "Barack Obama",
        "Joe Biden",
        "Joo E",
        "Hye Bin",
        "Jane",
        "Na Yun",
        "Ah In",
        "Nancy",
        "Taeha",
        "Daisy",
        "Yeon Woo"
    ]

    try:
        # Definition capture screen
        while True:
            # Print log information
            print(
                em.emojize(
                    '================================== Deep Learning - AiParkHub Organization (Feature Recognition)=> :satellite:  Training <Feature Model> :bar_chart:  ==========================',
                    use_aliases=True))

            # Grab a single frame of video
            ret, frame = resource_capture.read()

            # Settings window full screen
            cv2.namedWindow(
                'Deep Learning - AiParkHub Organization (Feature Recognition)',
                cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'Deep Learning - AiParkHub Organization (Feature Recognition)',
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations)

            # Loop through each face in this frame of video
            for (
                    top, right, bottom, left), face_encoding in zip(
                    face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)

                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to
                # the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw a box around the face
                cv2.rectangle(
                    frame, (left, top), (right, bottom), (255, 0, 0), 1)

                # Portrait Face Statistics
                cv2.putText(frame, "Feature Counts : " + str(len(face_locations)),
                            (20, 100), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

                cv2.putText(frame, "Stage Name : " + str(name),
                            (20, 150), font, 1, (0, 255, 0), 3, cv2.LINE_AA)

                print(
                    em.emojize(
                        'Feature Recognition info => :chart_with_upwards_trend:',
                        use_aliases=True),
                    f' DataSet[{top}, {right}, {bottom}, {left}] - Feature Counts => ' +
                    str(
                        len(face_locations)),
                    f', Stage Name[{name}]')

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            font, 1.0, (255, 255, 255), 2)

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
                (255,
                 255,
                 255),
                2,
                cv2.LINE_AA)
            cv2.putText(frame, "Q: Quit Training Model", (20, 750),
                        font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # '''
            # Visual information Style - 2 For 4K
            # '''
            # cv2.putText(frame, "Deep Learning - AiParkHub Organization (Feature Recognition)", (20, 40), font, 1.5,
            #             (240, 237, 232), 3, cv2.LINE_AA)
            # cv2.putText(frame, "Q: Quit Training Model", (20, 950), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Perform resource write
            resourcesWriter.write(frame)

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
    except Exception as e:
        print('Exception => ', e)
    finally:
        # Release resources
        resource_capture.release()
        cv2.destroyAllWindows()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    # call function
    fast_training_model(RESOURCE_PATH_0)
    # slow_training_model(RESOURCE_PATH_0)
