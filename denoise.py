import streamlit as st
import tensorflow as tf
import cv2
import time

# Define your custom lambda function outside the load_model function
def custom_lambda(x):
    # Define your custom lambda function here
    return x

# Add any other custom layers or functions here
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

# Ensure all custom objects are defined
custom_objects = {
    'TFOpLambda': tf.keras.layers.Lambda,
    'tf': tf,
    'custom_lambda': custom_lambda,
    'CustomLayer': CustomLayer  # Add your custom layer here
}

@st.cache_resource
def load_model(model_choice):
    model = tf.keras.models.load_model(f'{model_choice.lower()}.h5', compile=False, custom_objects=custom_objects)
    model.compile(optimizer='adam', loss='mse')
    return model

def get_patches(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    patch_size = 40
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches

def PSNR(ground_truth, denoised_image):
    # Implement the PSNR calculation here
    return 30.0  # Placeholder value

def predict_fun(model, patches_noisy, gt):
    # Implement the prediction function here
    return patches_noisy  # Placeholder value

def prediction_ui(gt, model_choice):
    st.sidebar.title('Denoising Options')
    noise_level = st.sidebar.slider('Noise Level', 10, 50, 10)
    model_filesize = 1.0  # Placeholder value

    if noise_level >= 10:
        patches_noisy = get_patches(gt)
        submit = st.sidebar.button('Predict Now')
        if submit:
            start = time.time()
            model = load_model(model_choice)
            denoised_image = predict_fun(model, patches_noisy, gt)
            end = time.time()
            
            st.header(f'Denoised image using {model_choice} model')
            st.markdown('( Model size: `%.3f` MB ) ( Time taken for prediction: `%.3f` seconds )' % (model_filesize, (end - start)))
            st.image(denoised_image)     
            st.success('PSNR of denoised image : %.3f db' % (PSNR(gt, denoised_image)))
    else:
        st.error("Choose a minimum noise level of 10...")

def main():
    st.title('Image Denoising Application')
    model_choice = st.sidebar.selectbox('Choose Model', ['DnCNN', 'RIDNet'])
    gt = None  # Placeholder for ground truth image
    prediction_ui(gt, model_choice)

if __name__ == '__main__':
    main()