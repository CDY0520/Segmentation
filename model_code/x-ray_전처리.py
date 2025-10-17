import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 1) 데이터 로드
IMAGE_DIR = 'Chest-X-Ray/image'
MASK_DIR = 'Chest-X-Ray/mask'
image_paths = sorted(glob(os.path.join(IMAGE_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.png')))

imgs = []
masks = []
for img_path, mask_path in zip(image_paths, mask_paths):
    img = img_to_array(load_img(img_path, color_mode='grayscale', target_size=(128,128))) / 255.0
    mask = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=(128,128))) / 255.0
    imgs.append(img)
    masks.append(mask)

X = np.array(imgs)   # shape: (N,128,128,1)
y = np.array(masks)  # shape: (N,128,128,1)

np.save('X_images.npy', X)
np.save('y_masks.npy', y)
