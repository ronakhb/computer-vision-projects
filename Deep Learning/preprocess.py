'''
preprocess.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains the code for preprocessing the images. 

'''
import os
from PIL import Image

# Function to process images
def process_image(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image to 128x128
    image = image.resize((128, 128))
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    
    # Invert the colors
    inverted_image = Image.eval(grayscale_image, lambda x: 255 - x)
    
    # Get the file name without extension
    # file_name, file_extension = os.path.splitext(image_path)
    
    # Save the inverted image with the same name
    inverted_image.save(image_path)

def main():
    # Folder containing the images
    folder_path = "datasets/greek_train/greek_train/psi"

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Iterate over the image files and process each image
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)
        
        # Process the image
        process_image(image_path)

    print("Images processed and saved successfully.")

if __name__ == "__main__":
    main()