# -*- coding:utf-8 -*-
#
# Geek International Park | 极客国际公园
# GeekParkHub | 极客实验室
# Website | https://www.geekparkhub.com
# Description | Open · Creation |
# Open Source Open Achievement Dream, GeekParkHub Co-construction has never been seen before.
# HackerParkHub | 黑客公园
# Website | https://www.hackerparkhub.org
# Description | In the spirit of fearless exploration, create unknown technology and worship of technology.
# GeekDeveloper : JEEP-711
#
# @Author : system
# @Version : 0.2.5
# @Program : 训练 人像特征模型 | Training portrait feature models
# @File : training_portrait_feature_models.py
# @File : training_feature_model.py
# @Description : 训练 人像特征模型 | Training portrait feature models


# 导入 第三方模块 | Import third-party modules
import cv2
import emoji as em
import numpy as np

# Draw font style
font = cv2.FONT_ITALIC

'''
定义 静态资源 | Definition static resource
'''
# Define training dataset
VIDEO_PATH_0 = 'aiparkhub/hamilton_clip.mp4'
VIDEO_PATH_1 = 'core/data/data_faces_from_camera/part_1/you2go.me-LuisFonsi-Despacito(BeatboxCover)-Hiss&Bigman-Shoutout.mp4 '
VIDEO_PATH_2 = 'core/data/data_faces_from_camera/part_1/Despacito-Beatbox-Cover-Goldfish-Brain-feat-Hiss.mp4'
VIDEO_PATH_3 = 'core/data/data_faces_from_camera/part_3/Fancam_BAAM.mp4'
VIDEO_PATH_4 = 'core/data/data_faces_from_camera/part_4/TensorFlow-high-level-APIs-going_deep_on_data_and_features.mp4'
VIDEO_PATH_5 = 'core/data/data_faces_from_camera/part_5/CheapThrills-MayJLee.flv'
VIDEO_PATH_6 = 'core/data/data_faces_from_camera/part_6/Fox-BoALiaKimChoreography.flv'
VIDEO_PATH_7 = 'core/data/data_faces_from_camera/part_7/Samsara-Tungevaag&Raaban-TinaBooChoreography.flv'
VIDEO_PATH_8 = 'core/data/data_faces_from_camera/part_8/Sugar-Maroon5-LiaKimChoreography.flv'
VIDEO_PATH_9 = 'core/data/data_faces_from_camera/part_9/ThisIsWhatYouCameFor-MayJLee.flv'
VIDEO_PATH_10 = 'core/data/data_faces_from_camera/part_10/Touch-LittleMix-MayJLeeChoreography.flv'
VIDEO_PATH_11 = 'core/data/data_faces_from_camera/part_11/MOMOLAND(모모랜드)-BAAM&BBoomBBoom(뿜뿜)@MAMA2018inHONGKONG.flv'
VIDEO_PATH_12 = 'core/data/data_faces_from_camera/part_12/TWICE-FANCY-MV.flv'
VIDEO_PATH_13 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/Programmers_talk.mp4'
VIDEO_PATH_14 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/28456156-1-64.flv'
VIDEO_PATH_15 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/25066902-1-64.flv'
VIDEO_PATH_16 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/TWICE-MOVE.flv'
VIDEO_PATH_17 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/0000.mp4'
VIDEO_PATH_18 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/132249458_nb2-1.flv'
VIDEO_PATH_19 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/141068232-1.flv'
VIDEO_PATH_20 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/RPReplay_Final_1.mp4'
VIDEO_PATH_21 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/RPReplay_Final_2.mp4'
VIDEO_PATH_22 = '/Volumes/SYSTEMHUB-711/private_other_flow/004/mac-data/RPReplay_Final_3.mp4'
VIDEO_PATH_23 = 'core/data/data_faces_from_camera/part_14/RPReplay_Final_4.mp4'
VIDEO_PATH_24 = 'core/data/data_faces_from_camera/part_15/training_132249458.mp4'
VIDEO_PATH_25 = 'core/data/data_faces_from_camera/part_15/training_141068232.mp4'


# 定义 训练模型函数 | Define training model functions
def training_model(training_dataset):
    '''
    Facial recognition in portrait areas using portrait feature cascade classifier engine
    使用 人像特征 级联分类器引擎 在人像区域进行面部识别
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    '''
    Eye recognition using human eyes cascade classifier engine
    使用 人眼 级联分类器引擎 在人像面部区域进行眼部识别
    '''
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    '''
    Eye Recognition in the Face Area of ​​a Portrait Using the Smile Cascade Classifier Engine
    使用 微笑 级联分类器引擎 在人像面部区域进行眼部识别
    '''
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Calling device
    cap = cv2.VideoCapture(training_dataset)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 1

    videoWriter = cv2.VideoWriter('core/data/data_faces_from_camera/part_15/test_data/test_training_part_2.mp4',
                                  cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60, (w, h))

    res = cap.isOpened()
    print('Check Device Status is =>', res)
    try:
        # Definition capture screen
        while res:
            print(
                em.emojize(
                    '================================== Deep Learning - AiParkHub Organization (Feature Recognition)=> :satellite:  Training <Feature Model> :bar_chart:  ==========================',
                    use_aliases=True))
            # Read frames
            ret, frame = cap.read()
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=1,
                                                  minSize=(25, 25))  # 1.3, 5 , minSize=(25, 25)

            # Settings window full screen
            cv2.namedWindow('Deep Learning - AiParkHub Organization (Feature Recognition)', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Deep Learning - AiParkHub Organization (Feature Recognition)',
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

            # Settings local window
            # cv2.namedWindow("Face Recognition", 0)
            # cv2.resizeWindow("Face Recognition", 1920, 1080)
            # cv2.moveWindow("Face Recognition", 350, 150)

            img = frame
            for (x, y, w, h) in faces:

                # Draw portrait face logo frame
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)

                # Portrait Face Statistics (20, 100), (1600, 100), 
                cv2.putText(img, "Feature-X Counts : " + str(len(faces)), (20, 100), font, 1, (0, 255, 255), 3,
                            cv2.LINE_AA)

                # cv2.rectangle(img, (x, y), (x+w+80, y+h-80), (0, 255, 255), cv2.FILLED)

                cv2.putText(img, f'Feature-X => DataSet [{x}, {y}, {w}, {h}]', (x - 5, y - 7), font, 1, (0, 255, 255),
                            2, cv2.LINE_AA)
                # cv2.putText(img, 'Faces', (x, y - 7), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                print(em.emojize('Feature-X Recognition info => :brain:', use_aliases=True),
                      f' DataSet[{x}, {y}, {w}, {h}] - Feature-X Counts => ' + str(len(faces)))
                # 在有效区域内识别人像面部, 节省计算资源
                face_area = img[y:y + h, x:x + w]
                # face_area = img

                # Portrait eye detection
                '''
                Use the human eye cascade classifier engine for eye recognition in the face area of ​​the portrait, and the return value is a list of eyes coordinates
                使用 人眼 级联分类器引擎 在人像面部区域进行眼部识别, 返回值为eyes坐标列表
                '''
                eyes = eye_cascade.detectMultiScale(face_area, 1.3, 10)

                for (ex, ey, ew, eh) in eyes:
                    # Draw portrait eye logo frame
                    cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                    # Portrait Eye Statistics, (1600, 150)
                    cv2.putText(img, "Feature-Y Counts : " + str(len(eyes)), (20, 150), font, 1, (0, 255, 0), 3,
                                cv2.LINE_AA)  # (20, 150)
                    cv2.putText(img, f'Feature-Y DataSet => [{ex}, {ey}, {ew}, {eh}]', (ex - 5, ey - 7), font, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(img, 'Eyes', (x + 60, y + 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print(em.emojize('Feature-Y Recognition info => :large_orange_diamond:', use_aliases=True),
                          f' DataSet[{ex}, {ey}, {ew}, {eh}] - Feature-Y Counts => ' + str(len(eyes)))

                # Portrait face smile detection
                '''
                Use the smile cascade classifier engine for eye recognition in the face area of ​​the portrait, and the return value is a list of eyes coordinates
                使用 微笑 级联分类器引擎 在人像面部区域进行眼部识别, 返回值为eyes坐标列表
                '''
                smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)

                for (ex, ey, ew, eh) in smiles:
                    # Draw portrait face smile identification box (255, 0, 0)
                    cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (240, 237, 232), 3)
                    # Portrait Smile Statistics, (1600, 200)
                    cv2.putText(img, "Feature-N Counts : " + str(len(smiles)), (20, 200), font, 1, (255, 0, 0), 3,
                                cv2.LINE_AA)  # (20, 200)
                    print(em.emojize('Feature-N Recognition info => :large_blue_diamond:', use_aliases=True),
                          f' DataSet[{ex}, {ey}, {ew}, {eh}] - Feature-N Counts => ' + str(len(smiles)))
                    cv2.putText(img, f'Feature-N DataSet => [{ex}, {ey}, {ew}, {eh}]', (ex - 5, ey - 7), font, 1,
                                (240, 237, 232), 2, cv2.LINE_AA)
                    # cv2.putText(img, 'Smile', (x + 60, y + 120), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            '''
            Visual information Style - 1
            '''
            # cv2.putText(img, "Deep Learning - AiParkHub (Feature Recognition)", (20, 40), font, 1.5,
            #             (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "N: Create Feature Folder", (20, 600), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "S: Save Current Feature", (20, 650), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "Q: Quit Training Model", (20, 700), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

            '''
            Visual information Style - 2
            '''
            # cv2.putText(img, "Deep Learning - AiParkHub Organization (Feature Recognition)", (350, 45), font, 1.2,
            #             (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "N: Create Feature Folder", (985, 600), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "S: Save Current Feature", (985, 650), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "Q: Quit Training Model", (985, 700), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

            '''
            Visual information Style - 3
            '''
            # cv2.putText(img, "Deep Learning - AiParkHub Organization (Feature Recognition)", (980, 45), font, 1.2, (240, 237, 232), 2,
            #             cv2.LINE_AA)
            # cv2.putText(img, "N: Create Feature Folder", (1620, 950), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "S: Save Current Feature", (1620, 1000), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "Q: Quit Training Model", (1620, 1050), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

            '''
            Visual information Style - 4 For Phone
            '''
            # cv2.putText(img, "Deep Learning - AiParkHub Organization (Feature Recognition)", (36, 270), font, 1, (240, 237, 232), 2,
            #             cv2.LINE_AA)
            # cv2.putText(img, "N: Create Feature Folder", (710, 1400), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "S: Save Current Feature", (710, 1450), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            # cv2.putText(img, "Q: Quit Training Model", (710, 1500), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

            '''
            Visual information Style - 5 For 4K
            '''
            cv2.putText(img, "Deep Learning - AiParkHub Organization (Feature Recognition)", (20, 40), font, 1.5,
                        (240, 237, 232), 3, cv2.LINE_AA)
            cv2.putText(img, "N: Create Feature Folder", (20, 1900), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            cv2.putText(img, "S: Save Current Feature", (20, 2000), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)
            cv2.putText(img, "Q: Quit Training Model", (20, 2100), font, 0.8, (240, 237, 232), 2, cv2.LINE_AA)

            videoWriter.write(frame)

            # Rendering effect
            cv2.imshow('Deep Learning - AiParkHub Organization (Feature Recognition)', img)
            # Monitor keyboard actions
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(
                    em.emojize(
                        '================================== Deep Learning - AiParkHub Organization (Feature Recognition) => :tada:  All Done! :tada:  ==========================',
                        use_aliases=True))
                break

    except Exception as e:
        print('Exception => ', e)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# 定义 主模块 | Definition Main module
if __name__ == '__main__':
    training_model(VIDEO_PATH_25)
