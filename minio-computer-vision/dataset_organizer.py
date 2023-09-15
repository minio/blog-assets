"""
Dataset organization script.
This script was inspired by similar code on this neat blog post: https://blog.paperspace.com/train-yolov5-custom-data/
"""
from sklearn.model_selection import train_test_split
import os
import shutil

def move_files_to_folder(files_list, destination_path):
    for f in files_list:
        try:
            shutil.move(f, destination_path)
        except:
            print(f)
            assert False

print("This script assumes you have collected all your sample images into a directory images/")
print("It also assumes you have collected all your YOLO-format annotation .txt files into a directory annotations/")

# Read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images')]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

# Use sklearn function to shuffle and split samples into train, val, and test sets.
train_images, val_images, train_annotations, val_annotations = train_test_split(images,
                                                                                annotations,
                                                                                test_size=0.2,
                                                                                random_state=42)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images,
                                                                              val_annotations,
                                                                              test_size=0.5,
                                                                              random_state=42)

# Move the image splits into their folders
move_files_to_folder(train_images, 'images/train/')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
# Move the annotation splits into their corresponding folders
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')
