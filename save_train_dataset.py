import os
from PIL import Image
import numpy as np

def resize_images(X, target_size):
    resized_images = []
    for image in X:
        img = Image.fromarray(image.astype(np.uint8))
        resized_img = img.resize((target_size, target_size))
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)


main_directory = "C:\Face recognition"
image_vectors = []
train_x = []
train_y = []

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith(".jpg"):

            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_rgb = image.convert("RGB")
            image_array = np.array(image_rgb)
            image_vectors.append(image_array)

            if "not_face" in root :
                train_y.append([0])
            else :
                train_y.append([1])

train_x = np.array(image_vectors)
train_y = np.array(train_y, dtype=int).T

train_x = resize_images(train_x, 64)

train_x = train_x.reshape(train_x.shape[0], -1).T
train_x = np.float32(train_x)
train_x = train_x / 255

print(train_x.shape)
print(train_y.shape)

np.savez_compressed('t.npz', train_x = train_x, train_y = train_y)

print("Done")