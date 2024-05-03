import os
import shutil
import cv2
import numpy as np


def process_images(folder_path, label, target_size=(256, 256)):
    images = []
    labels = []
    label_dict = {'bottle': 1, 'basket': 2, 'food': 3, 'cup': 4, 'jar': 5, 'can': 6, 'dish': 7, 'mug': 8, 'glass': 9}
    label_code = label_dict.get(label)  # Safely get the label code

    if label_code is None:
        print(f"Label '{label}' not found in label dictionary.")
        return None, None

    label_path = os.path.join(folder_path, str(label_code))  # Correct directory path

    if not os.path.exists(label_path):
        print(f"Directory not found: {label_path}")
        return None, None

    for img_name in os.listdir(label_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):  # Check for image files
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image from {img_path}")
                continue
            img_resized = cv2.resize(img, target_size)
            img_resized = img_resized.astype('float32') / 255.0
            img_flat = img_resized.flatten()
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


# processing images from each sub folder in the train, validation, and testing directories
train_bottle_images, train_bottle_labels = process_images("train", 'bottle')
train_basket_images, train_basket_labels = process_images('train', 'basket')
train_food_images, train_food_labels = process_images('train', 'food')
train_cup_images, train_cup_labels = process_images('train', 'cup')
train_jar_images, train_jar_labels = process_images('train', 'jar')
train_can_images, train_can_labels = process_images('train', 'can')
train_dish_images, train_dish_labels = process_images('train', 'dish')
train_mug_images, train_mug_labels = process_images('train', 'mug')
train_glass_images, train_glass_labels = process_images('train', 'glass')

val_bottle_images, val_bottle_labels = process_images("val", 'bottle')
val_basket_images, val_basket_labels = process_images('val', 'basket')
val_food_images, val_food_labels = process_images('val', 'food')
val_cup_images, val_cup_labels = process_images('val', 'cup')
val_jar_images, val_jar_labels = process_images('val', 'jar')
val_can_images, val_can_labels = process_images('val', 'can')
val_dish_images, val_dish_labels = process_images('val', 'dish')
val_mug_images, val_mug_labels = process_images('val', 'mug')
val_glass_images, val_glass_labels = process_images('val', 'glass')

test_bottle_images, test_bottle_labels = process_images("test", 'bottle')
test_basket_images, test_basket_labels = process_images('test', 'basket')
test_food_images, test_food_labels = process_images('test', 'food')
test_cup_images, test_cup_labels = process_images('test', 'cup')
test_jar_images, test_jar_labels = process_images('test', 'jar')
test_can_images, test_can_labels = process_images('test', 'can')
test_dish_images, test_dish_labels = process_images('test', 'dish')
test_mug_images, test_mug_labels = process_images('test', 'mug')
test_glass_images, test_glass_labels = process_images('test', 'glass')


# add processed images from each class into one numpy array for each dataset

train_images = np.concatenate(
    (train_bottle_images, train_basket_images, train_food_images, train_cup_images,
     train_jar_images, train_can_images, train_dish_images, train_mug_images, train_glass_images),
    axis=0)
train_labels = np.concatenate(
    (train_bottle_labels, train_basket_labels, train_food_labels, train_cup_labels,
     train_jar_labels, train_can_labels, train_dish_labels, train_mug_labels, train_glass_labels),
    axis=0)

val_images = np.concatenate(
    (val_bottle_images, val_basket_images, val_food_images, val_cup_images,
     val_jar_images, val_can_images, val_dish_images, val_mug_images, val_glass_images),
    axis=0)
val_labels = np.concatenate(
    (val_bottle_labels, val_basket_labels, val_food_labels, val_cup_labels,
     val_jar_labels, val_can_labels, val_dish_labels, val_mug_labels, val_glass_labels),
    axis=0)

test_images = np.concatenate(
    (test_bottle_images, test_basket_images, test_food_images, test_cup_images,
     test_jar_images, test_can_images, test_dish_images, test_mug_images, test_glass_images),
    axis=0)
test_labels = np.concatenate(
    (test_bottle_labels, test_basket_labels, test_food_labels, test_cup_labels,
     test_jar_labels, test_can_labels, test_dish_labels, test_mug_labels, test_glass_labels),
    axis=0)



