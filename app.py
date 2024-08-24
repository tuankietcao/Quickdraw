import train as tr
import torch
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt

# Load the model
model_path = "model5.pth"
model, _ = tr.load_model(model_path)
data_dir = 'Data'
class_mapping = tr.create_class_mapping(data_dir)
classes = {value: key for key, value in class_mapping.items()}

def newListRemove(element, list): 
  list.remove(element)
  return list
# Function to preprocess the image before feeding it into the model
def play_page():
    proba_check = st.sidebar.checkbox("See probabilities")
    def preprocess_image(img_data):
        img_array = np.asarray(img_data)
        img_array = cv2.resize(img_array, (28, 28))  # Resize to match your model's input size
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.reshape(img_array, (1, 28, 28))
        img_array = torch.from_numpy(img_array).float()
        img_array = img_array[np.newaxis, ...]  

        return img_array
    # Function to predict the class of the doodle
    def predict(image, threshold=0.5):
        if image is None:
            return None, None

        # Put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predictions = {classes[i]: prob.item() for i, prob in enumerate(probabilities) if prob.item() > threshold}
        
        # Sort predictions and show the top one


        return probabilities, predictions
    # Initialize session state variables if not already done
    if 'score' not in st.session_state:
        st.session_state.score = 0
        st.session_state.round = 1
        st.session_state.doodles = 6
        st.session_state.lives = 5
        st.session_state.current_class = random.choice(classes)
        st.session_state.next = False


    # Page decorations
    st.title("🎨 Quick, Draw! 🎨")

    def play_round():


        st.write(f"**Round {st.session_state.round}: Draw a {st.session_state.current_class}!**")
        st.write(f"**Lives: {'❤️' * st.session_state.lives}**")

        # Initialize canvas for drawing
        canvas_result = st_canvas(
                fill_color="white", 
                stroke_width=10,
                stroke_color="white",
                background_color="black", 
                height=500, 
                width=500,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )

        if canvas_result.image_data is not None and st.session_state.next is False:
            if st.session_state.lives <= 0 or st.session_state.doodles == 0:
                st.session_state.doodles -= 1
                st.session_state.round += 1
                st.session_state.current_class = random.choice(classes)
                st.session_state.lives = 5
                st.rerun()

            image = preprocess_image(canvas_result.image_data)
            proba, predictions = predict(image)

            if proba_check:

                col1, col2 = st.columns([1,4], vertical_alignment="center")
                with col1: 
                    st.image(canvas_result.image_data, width=150)
                with col2:
                    index = []
                    proba_plot = []
                    for i, prob in enumerate(proba):
                            index.append(i)
                            proba_plot.append(prob)
                    main_fig = plt.figure(figsize=(12,6))
                    ax = main_fig.add_subplot(111)
                    plt.barh(y=[classes[i] for i in index], width=proba_plot, color=["dodgerblue"]*4 + ["tomato"])
                    st.pyplot(main_fig, use_container_width=True)

            if predictions:
                    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
                    prediction_str = ", ".join([f"{name}" for name, _ in sorted_predictions])

                    if st.session_state.current_class in predictions:
                        st.write(f"**Oh, I know! It's a {st.session_state.current_class}!**")
                        st.balloons()
                        st.session_state.next = True

                        if st.button("Next"):
                            st.session_state.score += 1
                            st.session_state.doodles -= 1
                            st.session_state.round += 1
                            st.session_state.current_class = random.choice(classes)
                            st.session_state.lives = 5
                            st.session_state.next = False

                            st.rerun()

                    else: 
                        st.write(f"I see {prediction_str}...")
                        st.session_state.lives -= 1
        elif st.session_state.next is True:
            if st.button("Next"):
                    st.session_state.score += 1
                    st.session_state.doodles -= 1
                    st.session_state.round += 1
                    st.session_state.current_class = random.choice(classes)
                    st.session_state.lives = 5
                    st.session_state.next = False

                    st.rerun()


    # Start the game
    if st.session_state.doodles > 0:
        play_round()
    else:
        st.write(f"**Game Over! Your total score is {st.session_state.score} out of 6.**")
        if st.button("Play Again"):
            st.session_state.score = 0
            st.session_state.round = 1
            st.session_state.doodles = 6
            st.session_state.lives = 5
            st.session_state.current_class = random.choice(classes)
            st.session_state.next = False

            st.rerun()

def visualize_page():
        # Get a list of all class directories in the data directory
    data_dir = "images"
    class_dirs = [os.path.join(data_dir, c) for c in classes.values()]

    # Let the user select a class from the dropdown
    selected_class = st.sidebar.selectbox("Select Class to Visualize:", list(classes.values()))
    st.session_state.selected_class = selected_class

    if selected_class:
        class_dir = data_dir
        image_paths = [os.path.join(class_dir, f"{selected_class}_{i}.jpg") for i in range(0, 100)]  
        num_images = len(image_paths)

        # Create a grid layout to display images
        cols = 5  # Adjust the number of columns as needed
        rows = int(np.ceil(num_images / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 , rows))

        # Load and display images on the grid
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            axes.flat[i].imshow(image)
            axes.flat[i].axis("off")

        # Add a title to the visualization
        fig.suptitle(f"Class: {selected_class}", fontsize=16)
        st.pyplot(fig)

def main():
    pages = {"Play Game": play_page, "Visualize Data": visualize_page}
    selected_page = st.sidebar.selectbox("Select Page:", list(pages.keys()))
    pages[selected_page]()
    # play_page()

if __name__ == "__main__":
    main()