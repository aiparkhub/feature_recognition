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
# @Program : 训练 人像特征模型 | Training portrait feature models
# @File : training_portrait_feature_models.py
# @Description : 训练 人像特征模型 | Training portrait feature models
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.


# 导入 第三方模块 | Import third-party modules
import cv2
import emoji as em

# Draw font style
font = cv2.FONT_ITALIC

'''
Define training dataset
'''
COMMON_PATH = 'aiparkhub/core/data/feature_training_dataset/part_1/'
RESOURCE_PATH_0 = COMMON_PATH + 'momoland/dataset_for_video/momoland_fancam_baam.flv'


# 定义 训练模型函数 | Define training model functions
def training_model(training_dataset):
    '''
    Facial recognition in portrait areas using portrait feature cascade classifier engine
    使用 人像特征 级联分类器引擎 在人像区域进行面部识别
    '''
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml')

    '''
    Eye recognition using human eyes cascade classifier engine
    使用 人眼 级联分类器引擎 在人像面部区域进行眼部识别
    '''
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')

    '''
    Eye Recognition in the Face Area of ​​a Portrait Using the Smile Cascade Classifier Engine
    使用 微笑 级联分类器引擎 在人像面部区域进行眼部识别
    '''
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Calling device
    cap = cv2.VideoCapture(training_dataset)

    # Definition resource width and height
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 1

    # Recognize portrait faces from resources and output recognition results as new video files
    resourcesWriter = cv2.VideoWriter(
        'aiparkhub/core/data/feature_training_dataset/part_1/momoland/rendering_data/rendering_test_training_part_3.mp4',
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
            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.3, minNeighbors=5, minSize=(
                    25, 25))  # Defaults: 1.3, 3 , minSize=(25, 25)

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

            img = frame
            for (x, y, w, h) in faces:

                # Draw portrait face logo frame
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 255), 4)

                # Portrait Face Statistics
                cv2.putText(img, "Faces Counts : " + str(len(faces)),
                            (20, 100), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

                cv2.putText(
                    img,
                    f'Faces => DataSet [{x}, {y}, {w}, {h}]',
                    (x - 5,
                     y - 7),
                    font,
                    1,
                    (0,
                     255,
                     255),
                    2,
                    cv2.LINE_AA)
                print(em.emojize('Faces Recognition info => :brain:', use_aliases=True),
                      f' DataSet[{x}, {y}, {w}, {h}] - Faces Counts => ' + str(len(faces)))

                # 在有效区域内识别人像面部, 节省计算资源
                face_area = img[y:y + h, x:x + w]

                # Portrait eye detection
                '''
                Use the human eye cascade classifier engine for eye recognition in the face area of ​​the portrait, and the return value is a list of eyes coordinates
                使用 人眼 级联分类器引擎 在人像面部区域进行眼部识别, 返回值为eyes坐标列表
                '''
                eyes = eye_cascade.detectMultiScale(face_area, 1.3, 10)

                for (ex, ey, ew, eh) in eyes:
                    # Draw portrait eye logo frame
                    cv2.rectangle(
                        face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                    # Portrait Eye Statistics, (1600, 150)
                    cv2.putText(img, "Eyes Counts : " + str(len(eyes)),
                                (20, 150), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(
                        img,
                        f'Eyes DataSet => [{ex}, {ey}, {ew}, {eh}]',
                        (x + 60, y + 40),
                        font,
                        1,
                        (0,
                         255,
                         0),
                        2,
                        cv2.LINE_AA)
                    print(
                        em.emojize(
                            'Eyes Recognition info => :large_orange_diamond:',
                            use_aliases=True),
                        f' DataSet[{ex}, {ey}, {ew}, {eh}] - Eyes Counts => ' +
                        str(
                            len(eyes)))

                # Portrait face smile detection
                '''
                Use the smile cascade classifier engine for eye recognition in the face area of ​​the portrait, and the return value is a list of eyes coordinates
                使用 微笑 级联分类器引擎 在人像面部区域进行眼部识别, 返回值为eyes坐标列表
                '''
                smiles = smile_cascade.detectMultiScale(
                    face_area, scaleFactor=1.16, minNeighbors=65, minSize=(
                        25, 25), flags=cv2.CASCADE_SCALE_IMAGE)

                for (ex, ey, ew, eh) in smiles:
                    # Draw portrait face smile identification box
                    cv2.rectangle(
                        face_area, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)
                    # Portrait Smile Statistics, (1600, 200)
                    cv2.putText(img, "Smile Counts : " + str(len(smiles)),
                                (20, 200), font, 1, (255, 0, 0), 3, cv2.LINE_AA)
                    print(
                        em.emojize(
                            'Smile Recognition info => :large_blue_diamond:',
                            use_aliases=True),
                        f' DataSet[{ex}, {ey}, {ew}, {eh}] - Smile Counts => ' +
                        str(
                            len(smiles)))
                    cv2.putText(
                        img,
                        f'Smile DataSet => [{ex}, {ey}, {ew}, {eh}]',
                        (x + 60, y + 120),
                        font,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)

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

            # Rendering effect
            cv2.imshow(
                'Deep Learning - AiParkHub Organization (Feature Recognition)', img)
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
    # call function
    training_model(RESOURCE_PATH_0)
