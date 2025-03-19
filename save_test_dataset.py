import os
import numpy as np
from PIL import Image

def resize_images(X, target_size):
    resized_images = []
    for image in X:
        img = Image.fromarray(image.astype(np.uint8))
        resized_img = img.resize((target_size, target_size))
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

main_directory = "C:\Face Recognition\Face Recognition"
test_x = []
test_y = []

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith(".jpg"):

            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_rgb = image.convert("RGB")
            image_array = np.array(image_rgb)
            test_x.append(image_array)

            if "not_face" in root:
                test_y.append([0])
            else:
                test_y.append([1])

test_x = np.array(test_x)
test_y = np.array(test_y, dtype=int).T

test_x = resize_images(test_x, 64)

test_x = test_x.reshape(test_x.shape[0], -1).T
test_x = np.float32(test_x)
test_x = test_x / 255

print(test_x.shape)
print(test_y.shape)

np.savez_compressed('t1.npz', test_x = test_x, test_y = test_y)

print("Done")