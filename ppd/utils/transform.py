import cv2
import numpy as np
import torch
import torch.nn.functional as F

def image2tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image / 255.).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    return image

def video2tensor(images):
    tensor_list = []
    for i in range(images.shape[0]):
        image = images[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image / 255.).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        tensor_list.append(image)
    return tensor_list

def resize_1024(image):
    image = cv2.resize(image, (1024, 768), interpolation=cv2.INTER_LINEAR)
    return image

def resize_1024_crop(image):
    ori_h, ori_w = image.shape[:2]
    tar_w, tar_h = 1024, 768
    if ori_h > ori_w:
        resize_h = int(tar_w / ori_w * ori_h)
        image = cv2.resize(image, (tar_w, resize_h), interpolation=cv2.INTER_LINEAR)
        if resize_h > tar_h:
            top = (resize_h - tar_h) // 2
            image = image[top:top+tar_h, :]
        else:
            image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)

    else:
        resize_w = int(tar_h / ori_h * ori_w)
        image = cv2.resize(image, (resize_w, tar_h), interpolation=cv2.INTER_LINEAR)

        if resize_w > tar_w:
            left = (resize_w - tar_w) // 2
            image = image[:, left:left+tar_w]
        else:
            image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)

    return image

def resize_keep_aspect(image):
    ori_h, ori_w = image.shape[:2]
    tar_w, tar_h = 1024, 768
    ori_area = ori_h * ori_w
    tar_area = tar_h * tar_w
    scale = scale = (tar_area / ori_area) ** 0.5
    resize_h = ori_h * scale
    resize_w = ori_w * scale
    resize_h = max(16, int(round(resize_h / 16)) * 16)
    resize_w = max(16, int(round(resize_w / 16)) * 16)

    if scale < 1:
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
    return image





        



