import cv2
import csv
from os.path import join as pjoin
import skimage
import tensorflow as tf
import numpy as np
import os
import facenet
import align.detect_face

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

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

    # 如果检测出图片中不存在人脸 则直接返回，return 0（表示不存在人脸，跳过此图）
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

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(data_dir):
    # data为字典类型 key对应人物分类 value为读取的一个人的所有图片 类型为ndarray
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)       
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]         

        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics      
    return data


def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

# 模型位置
model_dir='./models/20170512-110547/'
with tf.Graph().as_default():
    with tf.Session() as sess:
        # 加载facenet模型
        facenet.load_model(model_dir)

        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # 从训练数据文件夹中加载图片并剪裁，最后embding，data为dict
        data=load_data('./train_dir/')
#         data=load_data('/home/python/notebook/taojinglong/deep-face/data/datasets/1216_new/train/')

        # keys列表存储图片文件夹类别（几个人）
        keys=[]
        for key in data:
            keys.append(key)
            print('folder:{},image numbers：{}'.format(key,len(data[key])))

        # 使用mtcnn模型获取每张图中face的数量以及位置，并将得到的embedding数据存储
        for n in range(len(keys)):
            for x in data[keys[n]]:
                _,images_me,i = load_and_align_data(x, 160, 44, 1.0)
                if i:
                    feed_dict = { images_placeholder: images_me, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict) 
                    for xx in range(len(emb)):
                        emb=list(emb[xx,:])
                        emb.append(keys[n])
                        with open('face_feature.csv', "a+", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(emb)
