import shutil
import cv2
import numpy as np
import os


def process_images(folder_path, label, target_size=(256, 256)):
    images = []
    labels = []
    label_dict = {'bottle': 1, 'basket': 2, 'food': 3, 'cup': 4, 'jar': 5, 'can': 6, 'dish': 7, 'mug': 8,
                  'glass': 9}  # Define a dictionary to map labels to numeric values
    label_code = label_dict[label]
    label_path = os.path.join(folder_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        # load the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        # resize the image
        img_resized = cv2.resize(img, target_size)  # Resize the image to the specified target size
        # preprocess the resized image (e.g., normalize pixel values)
        img_resized = img_resized.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        # flatten the resized image
        img_flat = img_resized.flatten()
        # append the flattened image and corresponding label
        images.append(img_flat)
        labels.append(label_code)
    return np.array(images), np.array(labels)


def split_data(source_folder, train_size=0.8, val_size=0.1, test_size=0.1):
    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    paths = {cls: [os.path.join(source_folder, cls, f) for f in os.listdir(os.path.join(source_folder, cls))] for cls in
             classes}

    # Creating training, validation, and test directories within each class directory
    for cls in classes:
        os.makedirs(os.path.join(source_folder, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(source_folder, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(source_folder, 'test', cls), exist_ok=True)

    # Shuffling and splitting the data
    for cls, images in paths.items():
        np.random.shuffle(images)
        n = len(images)
        train_end = int(train_size * n)
        val_end = train_end + int(val_size * n)
        test_images = images[val_end:]
        val_images = images[train_end:val_end]
        train_images = images[:train_end]

        # Copying images to respective folders
        for image in train_images:
            shutil.copy(image, os.path.join(source_folder, 'train', cls))
        for image in val_images:
            shutil.copy(image, os.path.join(source_folder, 'val', cls))
        for image in test_images:
            shutil.copy(image, os.path.join(source_folder, 'test', cls))


dataset_path = "data"
split_data(dataset_path)
