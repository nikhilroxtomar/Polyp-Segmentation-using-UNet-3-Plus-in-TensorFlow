
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import imageio
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import create_dir, load_dataset
from metrics import dice_loss, dice_coef

IMG_H = 512
IMG_W = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir(f"results")

    """ Load the model """
    model_path = os.path.join("files", "model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

    """ Dataset """
    dataset_path = "/media/nikhil/New Volume/ML_DATASET/Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Prediction """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_W, IMG_H))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        """ Read Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)

        """ Prediction """
        pred = model.predict(x, verbose=0)[0]
        pred = np.concatenate([pred, pred, pred], axis=-1)
        # pred = (pred > 0.5).astype(np.int32)

        """ Save final mask """
        line = np.ones((IMG_H, 10, 3)) * 255
        cat_images = np.concatenate([image, line, mask*255, line, pred*255], axis=1)
        save_image_path = os.path.join("results",  f"{name}.jpg")
        cv2.imwrite(save_image_path, cat_images)
