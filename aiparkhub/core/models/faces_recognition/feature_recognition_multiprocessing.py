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
# @Program : 多线程 特征识别 | Multi-threaded feature recognition
# @File : feature_recognition_multiprocessing.py
# @Description : 获取资源数据集进行人脸识别 | Obtaining a resource dataset for face recognition
# @Copyright © 2019 - 2020 AiParkHub-Organization. All rights reserved.


# 导入模块 | Import module
import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform
import emoji as em

# Draw font style
font = cv2.FONT_ITALIC

'''
Define training dataset
'''
COMMON_PATH = 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'president_obama/dataset_for_video/Former_President_Obama_unleashes_on_Trump_GOP_Full_speech_from_Illinois.mp4'
RESOURCE_PATH_1 = COMMON_PATH + 'president_obama&joe_biden/dataset_for_video/Obama_Tribute_to_Joe_Biden_(Full Speech).mp4'
RESOURCE_PATH_2 = COMMON_PATH + 'momoland/dataset_for_video/momoland_fancam_baam.flv'
RESOURCE_PATH_3 = COMMON_PATH + 'momoland/dataset_for_video/momoland_baam_japanese_ver.mp4'
RESOURCE_PATH_4 = COMMON_PATH + 'momoland/dataset_for_video/momoland_baam_special_video.mp4'
RESOURCE_PATH_5 = COMMON_PATH + 'momoland/dataset_for_video/momoland_baam_moving_dance_practice.mp4'
RESOURCE_PATH_6 = COMMON_PATH + 'momoland/dataset_for_video/momoland_bboom_bboom_japanese_ver_dance_video.mp4'


def next_id(current_id, worker_num):
    """
    Get next worker's id
    """

    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


def prev_id(current_id, worker_num):
    """
    Get previous worker's id
    """

    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


def capture(read_frame_list, Global, worker_num):
    """
    A subprocess use to capture frames.
    """

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(RESOURCE_PATH_2)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))
    
    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            # Settings window full screen
            # cv2.namedWindow('Deep Learning - AiParkHub Organization (Feature Recognition)', cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('Deep Learning - AiParkHub Organization (Feature Recognition)',
            #                       cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()


def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    """
    Many subprocess use to process frames.
    """

    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break
            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[:, :, ::-1]

        # Find all the faces and face encodings in the frame of video, cost most time
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame_process, (left, top), (right, bottom), (255, 0, 0), 3)

            # Portrait Face Statistics (20, 100), (1600, 100),
            cv2.putText(frame_process, "Feature Counts : " + str(len(face_locations)), (20, 100), font, 1,
                        (0, 255, 255), 3,
                        cv2.LINE_AA)

            cv2.putText(frame_process, "Stage Name : " + str(name), (20, 150), font, 1, (0, 255, 0), 3,
                        cv2.LINE_AA)

            print(em.emojize('Feature Recognition info => :chart_with_upwards_trend:', use_aliases=True),
                  f' DataSet[{top}, {right}, {bottom}, {left}] - Feature Counts => ' + str(len(face_locations)),
                  f', Stage Name[{name}]')

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

        '''
        Visual information Style - 1
        '''
        cv2.putText(frame_process, "Deep Learning - AiParkHub (Feature Recognition)", (20, 40), font, 1.5,
                    (240, 237, 232),
                    2, cv2.LINE_AA)
        cv2.putText(frame_process, "N: Create Feature Folder", (20, 950), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
        cv2.putText(frame_process, "S: Save Current Feature", (20, 1000), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
        cv2.putText(frame_process, "Q: Quit Training Model", (20, 1050), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

        # w = int(frame_process.get(cv2.CAP_PROP_FRAME_WIDTH)) + 1
        # h = int(frame_process.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 1

        videoWriter = cv2.VideoWriter(
        'aiparkhub/core/data/feature_training_dataset/part_1/momoland/rendering_data/rendering_test_training_part_1.mp4',
        cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60, (1920, 1080))

        videoWriter.write(frame_process)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)


# Definition Main module
if __name__ == '__main__':

    try:
        # Fix Bug on MacOS
        if platform.system() == 'Darwin':
            set_start_method('forkserver')

        # Global variables
        Global = Manager().Namespace()
        Global.buff_num = 1
        Global.read_num = 1
        Global.write_num = 1
        Global.frame_delay = 0
        Global.is_exit = False
        read_frame_list = Manager().dict()
        write_frame_list = Manager().dict()

        # Number of workers (subprocess use to process frames)
        if cpu_count() > 2:
            worker_num = cpu_count() - 1  # 1 for capturing frames
        else:
            worker_num = 2

        # Subprocess list
        p = []

        # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
        p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num)))
        p[0].start()

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
        jooe_face_encoding = face_recognition.face_encodings(momoland_jooe_photo)[0]

        # momoland_hyebin_photo
        momoland_hyebin_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_hyebin.jpg")
        hyebin_face_encoding = face_recognition.face_encodings(momoland_hyebin_photo)[0]

        # momoland_jane_photo
        momoland_jane_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_jane_v2.jpg")
        jane_face_encoding = face_recognition.face_encodings(momoland_jane_photo)[0]

        # momoland_nayun_photo
        momoland_nayun_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_nayun.jpg")
        nayun_face_encoding = face_recognition.face_encodings(momoland_nayun_photo)[0]

        # momoland_ahin_photo
        momoland_ahin_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_ahin_v2.jpg")
        ahin_face_encoding = face_recognition.face_encodings(momoland_ahin_photo)[0]

        # momoland_nancy_photo
        momoland_nancy_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_nancy_v3.png")
        nancy_face_encoding = face_recognition.face_encodings(momoland_nancy_photo)[0]

        # momoland_taeha_photo
        momoland_taeha_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_taeha_v2.png")
        taeha_face_encoding = face_recognition.face_encodings(momoland_taeha_photo)[0]

        # momoland_daisy_photo
        momoland_daisy_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_daisy_v2.jpg")
        daisy_face_encoding = face_recognition.face_encodings(momoland_daisy_photo)[0]

        # momoland_yeonwoo_photo
        momoland_yeonwoo_photo = face_recognition.load_image_file(
            COMMON_PATH + "momoland/dataset_for_photo/momoland_yeonwoo.jpg")
        yeonwoo_face_encoding = face_recognition.face_encodings(momoland_yeonwoo_photo)[0]

        # ==============================================================================================================

        # Create arrays of known face encodings and their names
        Global.known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding,
            yeonwoo_face_encoding,
            jooe_face_encoding,
            hyebin_face_encoding,
            jane_face_encoding,
            nayun_face_encoding,
            ahin_face_encoding,
            nancy_face_encoding,
            taeha_face_encoding,
            daisy_face_encoding
        ]

        Global.known_face_names = [
            "Barack Obama",
            "Joe Biden",
            "Yeon Woo",
            "Joo E",
            "Hye Bin",
            "Jane",
            "Na Yun",
            "Ah In",
            "Nancy",
            "Taeha",
            "Daisy"
        ]

        # Settings window full screen
        cv2.namedWindow('Deep Learning - AiParkHub Organization (Feature Recognition)', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Deep Learning - AiParkHub Organization (Feature Recognition)',
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Create workers
        for worker_id in range(1, worker_num + 1):
            p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num)))
            p[worker_id].start()

        # Start to show video
        last_num = 1
        fps_list = []
        tmp_time = time.time()
        while not Global.is_exit:
            while Global.write_num != last_num:
                last_num = int(Global.write_num)

                # Calculate fps
                delay = time.time() - tmp_time
                tmp_time = time.time()
                fps_list.append(delay)
                if len(fps_list) > 5 * worker_num:
                    fps_list.pop(0)
                fps = len(fps_list) / numpy.sum(fps_list)
                print("FPS => %.2f" % fps)

                # Calculate frame delay, in order to make the video look smoother.
                # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
                # Larger ratio can make the video look smoother, but fps will hard to become higher.
                # Smaller ratio can make fps higher, but the video looks not too smoother.
                # The ratios below are tested many times.
                if fps < 6:
                    Global.frame_delay = (1 / fps) * 0.75
                elif fps < 20:
                    Global.frame_delay = (1 / fps) * 0.5
                elif fps < 30:
                    Global.frame_delay = (1 / fps) * 0.25
                else:
                    Global.frame_delay = 0

                # Display the resulting image
                cv2.imshow('Deep Learning - AiParkHub Organization (Feature Recognition)',
                           write_frame_list[prev_id(Global.write_num, worker_num)])

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                Global.is_exit = True
                print(
                    em.emojize(
                        '================================== Deep Learning - AiParkHub Organization (Feature Recognition) => :tada:  All Done! :tada:  ==========================',
                        use_aliases=True))
                break
            time.sleep(0.01)
    except Exception as e:
        print('Exception => ', e)
    finally:
        # Quit
        cv2.destroyAllWindows()
