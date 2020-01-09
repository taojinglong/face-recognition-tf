# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os
import copy
import facenet
import align.detect_face
import skimage
from os.path import join as pjoin
import joblib
from PIL import Image,ImageFont,ImageDraw

# =================================
# 载入人脸数据库
face_feature_know_path = "./face_feature.csv"
face_feature_know = pd.read_csv(face_feature_know_path, header=None,encoding='gbk')
_face_f = face_feature_know.copy()
features_known_arr = np.array(_face_f.values)
print("数据库人脸数:", len(features_known_arr))

# =================================
# 人脸检测、处理相关
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    if dist > 0.45:
        return "no",dist
    else:
        return "yes",dist

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

# 中文输出
def new_paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('./noto/NotoSansCJK-Bold.ttc',28, encoding="utf-8")
    fillColor = color
    position = pos #(20,40)
    # 排序
    chinese = sorted(chinese.items(),key=lambda item:item[0])
    for i in chinese:
        chinese_ = i[-1]
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position,chinese_,font=font,fill=fillColor)
        img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        position = (position[0],position[1]+32)
    return img

# ==============================
# mtcnn 载入
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold 三步的阈值
factor = 0.709 # scale factor 比例因子
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 创建mtcnn网络，并加载参数
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

def load_and_align_data(image, image_size, margin, gpu_memory_fraction):
    # 读取图片 
    img = image
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return 0,0,0
    else:    
        crop=[]
        det=bounding_boxes
        det[:,0]=np.maximum(det[:,0], 0)
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])
        det=det.astype(int)
        for i in range(len(bounding_boxes)):
            temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
            aligned=skimage.transform.resize(temp_crop, (image_size, image_size))

            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image=np.stack(crop)
            
        return det,crop_image,1

# ==============================
# facenet 模型载入、人脸实时检测
# 模型位置
model_dir='./models/facenet/'#"Directory containing the graph definition and checkpoint files.")

with tf.Graph().as_default():
    with tf.Session() as sess:  
        # 加载模型
        facenet.load_model(model_dir)
        print('建立facenet embedding模型')
        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # 摄像头实时监测
        c=0
        font = cv2.FONT_HERSHEY_COMPLEX
        frame_interval=3 # frame intervals  
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(1)
        while cap.isOpened():
            kk = cv2.waitKey(1)
            ret, frame = cap.read()
            timeF = frame_interval
            detect_face=[]
            if(c%timeF == 0):
                find_results=[]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.ndim == 2:
                    img = to_rgb(gray)
                det,crop_image,j= load_and_align_data(img, 160, 44, 1.0)
                if j:
                    feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }        
                    emb = sess.run(embeddings, feed_dict=feed_dict) 
                    emb=list(emb[0,:])
                    for i in range(len(features_known_arr)):
                        compare = return_euclidean_distance(emb, features_known_arr[i][0:-1])
                        if compare[0]=="yes":
                            result=features_known_arr[i][-1]
                            print("result:",result)
                            break
                        else:
                            result="wait"
                    # 绘制矩形框并标注文字
                    for rec_position in range(len(det)):
                        cv2.rectangle(frame,(det[rec_position,0],det[rec_position,1]),(det[rec_position,2],det[rec_position,3]),
                        (0, 255, 0), 2, 8, 0)
                        cv2.putText(frame,result, 
                        (det[rec_position,0]+10,det[rec_position,1]),
                        font, 1, (0, 0 ,255), thickness = 2, lineType = 2)
                new_frame = np.zeros_like(frame)
                new_frame[:]=255
                # new_frame = frame.copy()
                # cv2.namedWindow("camera",cv2.WINDOW_NORMAL) # 可调整
                cv2.namedWindow('camera',cv2.WINDOW_AUTOSIZE) # 不可调整
                cv2.imshow('camera',frame)
                # cv2.moveWindow("camera",200,150)
 
            if kk == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        
