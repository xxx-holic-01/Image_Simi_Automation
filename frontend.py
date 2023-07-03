import streamlit as st
import os
import json
import shutil
import sys
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Set the paths
model_path = r"C:\Users\Lenovo\Desktop\Classification_Model\keras_model.h5"
labels_path = r"C:\Users\Lenovo\Desktop\Classification_Model\labels.txt"
des_path = f"C:/Users/Lenovo/Desktop/Classification_Model/des"
 
# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
class_names = open(labels_path, "r").readlines()

# Define functions to preprocess images and extract features
def preprocess_image(image):
    image = image.resize((224, 224))
    image = ImageOps.fit(image, (224, 224))
    image = image.convert("RGB")
    image = np.array(image)
    image = preprocess_input(image)
    return image

def extract_features(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    features = np.reshape(features, (features.shape[0], -1))
    return features

# Define the function to perform similarity matching
def find_similarity(input_image_path, folder_path):
    # Load the input image
    input_image = Image.open(input_image_path)

    # Extract features from the input image
    input_features = extract_features(input_image)

    folder_features = []
    image_paths = []

    # Loop through the images in the folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Exclude non-image files and JSON files
            if not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.png')):
                continue

            # Load and preprocess each image in the folder
            image = Image.open(file_path)
            features = extract_features(image)

            # Store the features and image paths for later use
            folder_features.append(features)
            image_paths.append(file_path)

    # Convert the lists to numpy arrays
    folder_features = np.concatenate(folder_features)
    image_paths = np.array(image_paths)

    # Compute cosine similarity scores
    similarity_scores = cosine_similarity(input_features, folder_features)

    # Find the indices of the top 20 most similar images
    top_indices = np.argsort(similarity_scores.ravel())[::-1][:20]

    # Get the paths of the top 20 most similar images
    top_image_paths = image_paths[top_indices]
    top_similarity_scores = similarity_scores[0, top_indices]

    return top_image_paths, top_similarity_scores

def main():
    st.title("Image Classification and Similarity Matching")
# Create the "uploads" directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary location
        image_path = os.path.join("uploads", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform image classification and similarity matching
        # Load the model
        with st.spinner("Running the code..."):

            model = load_model(model_path, compile=False)

            # Load the labels
            class_names = open(labels_path, "r").readlines()

            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Load the input image
            image = Image.open(image_path).convert("RGB")

            # Resize and normalize the image
            size = (224, 224)
            image = ImageOps.fit(image, size)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array

            # Predict the category of the input image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]

            # Print the predicted category
            st.subheader("Predicted Category:")
            st.write(class_name[2:])

            # Check if the user wants to add the image to the predicted category folder
            if st.button("Add Image to Predicted Category"):
                shutil.copy(image_path, os.path.join(des_path, class_name[3:-1]))

            # Check if the user wants to create a new category manually for this image
            if st.button("Create New Category"):
                new_category = st.text_input("Enter the name for the new category:")
                if st.button("Create Category"):
                    # Create the new folder in the destination directory
                    new_folder_path = os.path.join(des_path, new_category)
                    os.makedirs(new_folder_path, exist_ok=True)
                    shutil.copy(image_path, new_folder_path)
                    st.success("New category created and image added.")
            if st.button("continue"):
                pass
            # Select the folder based on the predicted category
            folder_path = os.path.join(des_path, class_name[3:-1])
    
            # Perform similarity matching
            top_image_paths, top_similarity_scores = find_similarity(image_path, folder_path)

        # Display the results
        st.subheader("Top 20 Similar Images")
        for image_path, similarity_score in zip(top_image_paths, top_similarity_scores):
            st.image(Image.open(image_path))
            st.write(f"Similarity Score: {similarity_score:.4f}")

        # Delete the uploaded image from the temporary location
        os.remove(image_path)


if __name__ == "__main__":
    main()
