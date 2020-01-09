import os
import cv2
import numpy as np
from cv2_face_detector import FaceDetector

# OpenCv 调用摄像头 use camera
cap = cv2.VideoCapture(0)
cnt_ss = 0
# 存储人脸的文件夹
current_face_dir = ""

# 保存 faces images 的路径
path_photos_from_camera = "./faces_from_camera/"
# 新建保存人脸图像文件
def pre_work_mkdir():
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)
pre_work_mkdir()

# 如果有之前录入的人脸 
# 在之前 person_x 的序号按照 person_x+1 开始录入
if os.listdir("./faces_from_camera/"):
    # 获取已录入的最后一个人脸序号 / get the num of latest person
    person_list = os.listdir("./faces_from_camera/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)
# 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入
else:
    person_cnt = 0

# 之后用来控制是否保存图像的 flag
save_flag = 1
# 之后用来检查是否先按 'n' 再按 's'
press_n_flag = 0
# 使用ssd 人脸检测器
faceDetector = FaceDetector('ssd', 0.5)
# 字体类型
font = cv2.FONT_HERSHEY_COMPLEX
while cap.isOpened():
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)
    frame = img_rd.copy()
    faces = faceDetector.detect(frame)
    # 固定截取窗口
    (x,y,w,h) = (150,100,250,250)
    cv2.rectangle(frame, (x, y),(x+w, y+h),
                    (0, 0, 255) , 2)

    # press 'n' to create the folders for saving faces
    if kk == ord('n'):
        person_cnt += 1
        current_face_dir = path_photos_from_camera + "person_" + str(person_cnt)
        os.makedirs(current_face_dir)
        print('\n')
        print("新建的人脸文件夹 / Create folders: ", current_face_dir)
        cnt_ss = 0              # clear the cnt of faces
        press_n_flag = 1        # have pressed 'n'

    # 检测到人脸
    if len(faces) != 0:
        print(len(faces))
        for rect in faces:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y),(x+w, y+h),
                            (0, 255, 0) , 2)
            if save_flag:
                # press 's' to save faces into local images
                if kk == ord('s'):
                    # check if you have pressed 'n'
                    if press_n_flag:
                        cnt_ss += 1
                        # 保存固定窗口截图
                        im_blank = img_rd[100:350,150:400,:]
                        cv2.imwrite(current_face_dir + '/' +str("%03d" % cnt_ss) + ".jpg", im_blank)
                        print("写入本地 / Save into：", str(current_face_dir)+ '/' + str("%03d" % cnt_ss) + ".jpg")
                    else:
                        print("请在按 'S' 之前先按 'N' 来建文件夹 / Please press 'N' before 'S'")

    # # 添加说明 / add some statements
    cv2.putText(frame, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # show the numbers of faces detected
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    # press 'q' to exit
    if kk == ord('q'):
        break
    # 如果需要摄像头窗口大小可调
    cv2.namedWindow("camera", 0)
    cv2.imshow("camera", frame)

# 释放摄像头 / release camera
cap.release()
cv2.destroyAllWindows()