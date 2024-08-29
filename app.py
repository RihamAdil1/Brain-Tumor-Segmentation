import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

# Load the model
model_path = '/home/riham/Brain-Tumor-Segmentation/random_forest_model.pkl'  
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Function to extract features from the image
def extract_features(image_path):
    # Read the image
    input_img = np.array(Image.open(image_path))

    # Convert to grayscale if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    # Create a DataFrame to store features
    temp_df = pd.DataFrame()

    # Feature extraction code (same as in the training phase)
    pixel_values = img.reshape(-1)
    temp_df['Pixel_Value'] = pixel_values

    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    kernel = cv2.getGaborKernel((9, 9), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    temp_df[gabor_label] = filtered_img
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1
    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1)
    temp_df['Canny Edge'] = edges1

    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    temp_df['Roberts'] = edge_roberts1

    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    temp_df['Sobel'] = edge_sobel1

    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    temp_df['Scharr'] = edge_scharr1

    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    temp_df['Prewitt'] = edge_prewitt1

    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    temp_df['Gaussian s3'] = gaussian_img1

    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    temp_df['Gaussian s7'] = gaussian_img3

    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    temp_df['Median s3'] = median_img1

    return temp_df

# Streamlit app
st.set_page_config(page_title="Brain Tumor Segmentation App", page_icon=":brain:", layout="wide")

st.title("Brain Tumor Segmentation App")

# Display description with larger text
st.markdown("""
    <div style="font-size:24px;">
    **Welcome to the Brain Tumor Segmentation App!**
    <br><br>
    This app is designed by Rihal for **XX Hospital**, where all patients can upload their MRI images to monitor the size of their brain tumor.
    It helps in assessing whether the tumor is responding to chemotherapy and tracks the progress over time.
    </div>
""", unsafe_allow_html=True)

# Display company logo with smaller size
st.image("/home/riham/Brain-Tumor-Segmentation/logo.jfif", caption="Rihal", use_column_width=False, width=100)  # Adjust width as needed

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    img_width, img_height = image.size

    # Save the uploaded image to a temporary file for processing
    temp_image_path = "/tmp/uploaded_image.png"
    image.save(temp_image_path)

    # Extract features from the image
    features = extract_features(temp_image_path)

    # Make predictions using the trained model
    predicted_mask = model.predict(features)
    predicted_mask = predicted_mask.reshape((img_height, img_width))  # Reshape using image dimensions

    # Plot the segmented image with fixed size
    fig_segmented, ax_segmented = plt.subplots(figsize=(8, 8))  # Adjust figsize as needed
    ax_segmented.imshow(predicted_mask, cmap="gray")
    ax_segmented.set_title("Segmented Image")
    plt.axis('off')  # Hide axis
    fig_segmented.tight_layout()

    # Create columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)  # Ensure the uploaded image is displayed at its original size

    with col2:
        st.pyplot(fig_segmented)  # Display the segmented image next to the uploaded image
