from yolort.utils import get_image_from_url
import cv2
import numpy as np

in_size = 640
img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/bus.jpg"
# img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/zidane.jpg"
img = get_image_from_url(img_source)

img = img.astype("float32")
img = cv2.resize(img, (in_size, in_size))

img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)
print(img.shape)