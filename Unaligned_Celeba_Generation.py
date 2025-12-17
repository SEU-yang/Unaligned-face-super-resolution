from tqdm import tqdm
import os
import scipy
import scipy.ndimage
import numpy as np
from PIL import Image
import random
import cv2
import mediapipe as mp

# ================== 全局参数 ==================
output_size = 256
transform_size = 256
max_rotation = 50  # ±旋转角度
min_face_span_ratio = 0.4  # 最小脸占比
global_skipped_small_faces = 0  # 全局统计

# ================== MediaPipe Face Detection ==================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def face_detected(img_pil):
    """判断图像中是否有人脸"""
    img_rgb = np.array(img_pil)
    results = face_detection.process(img_rgb)
    return results.detections is not None

# ================== 加载 landmarks ==================
def load_landmarks(landmark_file):
    landmarks_dict = {}
    with open(landmark_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            landmarks = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(5, 2)
            landmarks_dict[filename] = landmarks
    return landmarks_dict

# ================== 对齐 + 随机旋转 + 小脸过滤 ==================
def align_face_with_random_rotation(img, landmarks, enable_padding=True, enable_random_rotation=True):
    global global_skipped_small_faces

    # 随机旋转 [-max_rotation, +max_rotation]
    if enable_random_rotation:
        # 修改为两个区间：
        if random.random() < 0.5:
            # 左旋 [-50, -25]
            angle = random.uniform(-30, -0)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, center=(img.width / 2, img.height / 2))
        else:
            # 右旋 [25, 50]
            angle = random.uniform(0, 30)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, center=(img.width / 2, img.height / 2))

        # angle = random.uniform(-max_rotation, max_rotation)
        # img = img.rotate(angle, resample=Image.Resampling.BILINEAR, center=(img.width / 2, img.height / 2))

    # 辅助向量
    eye_left = np.array(landmarks[0])
    eye_right = np.array(landmarks[1])
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left

    mouth_left = np.array(landmarks[3])
    mouth_right = np.array(landmarks[4])
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # 选择裁剪矩形
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 1.9, np.hypot(*eye_to_mouth) * 1.7)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # ================== 小脸过滤 ==================
    face_span = max(np.linalg.norm(eye_right - eye_left), np.linalg.norm(mouth_right - mouth_left))
    if face_span < output_size * min_face_span_ratio:
        global_skipped_small_faces += 1
        return None

    # Shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1])))
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1])
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1])))
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0)
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3])
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    # ================== 人脸检测过滤 ==================
    if not face_detected(img):
        return None

    return img

# ================== 批量处理 ==================
#def process_celeba_images(img_dir, landmark_file, save_dir):
    #os.makedirs(save_dir, exist_ok=True)
    #landmarks_dict = load_landmarks(landmark_file)
    #img_list = sorted(os.listdir(img_dir))

    #for img_name in tqdm(img_list):
        #if img_name not in landmarks_dict:
            #continue
        #img_path = os.path.join(img_dir, img_name)
        #img = Image.open(img_path).convert('RGB')
        #landmarks = landmarks_dict[img_name]
        #aligned_img = align_face_with_random_rotation(img, landmarks)
        #if aligned_img is None:
            #continue  # 小脸或人脸检测未通过
        #aligned_img.save(os.path.join(save_dir, img_name))
        
        
        
# ================== 批量处理 ==================
def process_celeba_images(img_dir, landmark_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    landmarks_dict = load_landmarks(landmark_file)
    img_list = sorted(os.listdir(img_dir))

    for img_name in tqdm(img_list):
        if img_name not in landmarks_dict:
            continue
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        landmarks = landmarks_dict[img_name]
        aligned_img = align_face_with_random_rotation(img, landmarks)
        if aligned_img is None:
            continue  # 小脸或人脸检测未通过
        
        # 分离文件名和扩展名
        name, ext = os.path.splitext(img_name)
        new_name = f"{name}_new{ext}"
        
        aligned_img.save(os.path.join(save_dir, new_name))
        

if __name__ == '__main__':
    img_dir = 'Celeba_Total/img_celeba.7z/img_celeba'
    landmark_file = 'Celeba_Total/Anno-20250426T074101Z-001/Anno/list_landmarks_celeba.txt'
    save_dir = 'aligned_random_rotated0030_end_1'

    process_celeba_images(img_dir, landmark_file, save_dir)
    print(f"Skipped small faces: {global_skipped_small_faces}")
