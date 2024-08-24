import os
import numpy as np
import cv2

def extract_images(data_dir, num_images):
    """
    Extracts a specified number of images from each .npy file in a directory and stores them in separate folders.

    Args:
        data_dir (str): The path to the directory containing the .npy files.
        num_images (int): The number of images to extract from each .npy file.
    """

    for class_dir in os.listdir(data_dir):
            print(class_dir)
            dir = os.path.join(data_dir, class_dir)
            data = np.load(dir)
            output_dir = "images"

            num_image = data.shape[0]
            data = data.reshape(num_image, 28, 28)

                # Extract a random sample of images
            indices = np.random.choice(num_image, num_images, replace=False)
            extracted_images = data[indices]
            print(extracted_images.shape)

                # Save the extracted images as individual files
            for i, image in enumerate(extracted_images):
                    image_path = os.path.join(output_dir, f"{class_dir.split('.')[0]}_{i}.jpg")
                    cv2.imwrite(image_path, image)

# Example usage
data_dir = "Data"
num_images = 100
extract_images(data_dir, num_images)