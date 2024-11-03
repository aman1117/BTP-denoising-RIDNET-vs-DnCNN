import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import ssl
import urllib

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('About the Project', 'Evaluate the model', 'View source code')
    )
    
    readme_text = st.markdown(get_file_content_as_string("README.md"))
    
    if selected_box == 'About the Project':
        st.sidebar.success('To try by yourself select "Evaluate the model".')
    if selected_box == 'Evaluate the model':
        readme_text.empty()
        evaluate_model()
    if selected_box == 'View source code':
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))

@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/sunilbelde/Imagedenoising-dncnn-keras/master/' + path
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)
    return response.read().decode("utf-8")

def evaluate_model():
    st.title('Denoise your image with DnCNN or RIDNET model')
    
    choice = st.sidebar.selectbox("Choose how to load image", ["Use Existing Images", "Browse Image"])
    model_choice = st.sidebar.radio("Choose a model to predict", ('DnCNN', 'RIDNET'), 0)
    
    if choice == "Browse Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image file", type="jpg")

        if uploaded_file is not None:
            # Convert the file to an OpenCV image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            gt = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            prediction_ui(gt, model_choice)
          
    if choice == "Use Existing Images":
        image_file_chosen = st.sidebar.selectbox('Select an existing image:', get_list_of_images(), 10)
        
        if image_file_chosen:
            images_path = os.path.join(os.getcwd(), 'images')
            gt = cv2.imread(os.path.join(images_path, image_file_chosen))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            prediction_ui(gt, model_choice)

def get_list_of_images():
    file_list = os.listdir(os.path.join(os.getcwd(), 'images'))
    filenames = sorted([str(filename) for filename in file_list if str(filename).endswith('.jpg')])
    return filenames

def prediction_ui(gt, model_choice):
    models_load_state = st.text('\n Loading model...')
    model = load_model(model_choice)
    models_load_state.text('\n Model Loading complete')

    model_filesize = os.stat(f'{model_choice.lower()}.tflite' if model_choice == 'RIDNET' else f'{model_choice.lower()}.h5').st_size / (1024 * 1024)
    
    noise_level = st.sidebar.slider("Pick the noise level", 10, 45, 25)
    
    ground_truth, noisy_image, patches_noisy = get_image(gt, noise_level=noise_level)
    st.header('Input Image')
    st.markdown('** Noise level : ** `%d`  (Noise level `0` will be the same as the original image)' % noise_level)
    st.image(noisy_image)
    
    if noise_level >= 10:
        st.success('PSNR of Noisy image : %.3f db' % PSNR(ground_truth, noisy_image))
      
        submit = st.sidebar.button('Predict Now')
        if submit:
            start = time.time()
            denoised_image = predict_fun(model, patches_noisy, gt, model_choice)
            end = time.time()
            
            st.header(f'Denoised image using {model_choice} model')
            st.markdown('( Model size: `%.3f` MB ) ( Time taken for prediction: `%.3f` seconds )' % (model_filesize, (end - start)))
            st.image(denoised_image)     
            st.success('PSNR of denoised image : %.3f db' % (PSNR(ground_truth, denoised_image)))
    else:
        st.error("Choose a minimum noise level of 10...")

@st.cache_resource
def load_model(model_choice):
    if model_choice == 'RIDNET':
        interpreter = tf.lite.Interpreter(model_path=f'{model_choice.lower()}.tflite')
        interpreter.allocate_tensors()
        return interpreter
    else:
        def custom_lambda(x):
            # Define your custom lambda function here
            return x

        custom_objects = {
            'TFOpLambda': tf.keras.layers.Lambda,
            'tf': tf,
            'custom_lambda': custom_lambda  # Add your custom lambda function here
        }
        
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
            x = image[i:i + patch_size, j:j + patch_size]
            patches.append(x)
    return np.array(patches)

def get_image(gt, noise_level):
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)  # Convert back to BGR for consistency
    noise = np.random.normal(0, noise_level, gt.shape)
    noisy_image = np.clip(gt + noise, 0, 255).astype(np.uint8)
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    patches_noisy = get_patches(noisy_image)
    return gt, noisy_image, patches_noisy

def predict_fun(model, patches_noisy, gt, model_choice):
    patch_size = 40
    denoised_image = np.zeros_like(gt, dtype=np.float32)
    total_patches = (denoised_image.shape[0] // patch_size) * (denoised_image.shape[1] // patch_size)
    progress_bar = st.progress(0)

    if model_choice == 'RIDNET':
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        index = 0
        for i in range(0, denoised_image.shape[0] - patch_size + 1, patch_size):
            for j in range(0, denoised_image.shape[1] - patch_size + 1, patch_size):
                patch = patches_noisy[index].astype('float32') / 255.0
                patch = np.expand_dims(patch, axis=0)

                model.set_tensor(input_details[0]['index'], patch)
                model.invoke()
                denoised_patch = model.get_tensor(output_details[0]['index'])[0] * 255.0

                denoised_image[i:i + patch_size, j:j + patch_size] += denoised_patch
                index += 1

                # Update progress bar
                progress_bar.progress(index / total_patches)

    else:  # DnCNN
        patches_noisy = patches_noisy.astype('float32') / 255.
        denoised_patches = model.predict(patches_noisy) * 255.0

        index = 0
        for i in range(0, denoised_image.shape[0] - patch_size + 1, patch_size):
            for j in range(0, denoised_image.shape[1] - patch_size + 1, patch_size):
                denoised_image[i:i + patch_size, j:j + patch_size] += denoised_patches[index]
                index += 1

                # Update progress bar
                progress_bar.progress(index / total_patches)

    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    progress_bar.empty()  # Remove the progress bar after completion
    return denoised_image

def PSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

if __name__ == "__main__":
    main()
