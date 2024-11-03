import streamlit as st
import cv2      
import os
import urllib
import numpy as np    
import tensorflow as tf
import time
import ssl

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
        models()
    if selected_box == 'View source code':
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))

@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/sunilbelde/Imagedenoising-dncnn-keras/master/' + path
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)
    return response.read().decode("utf-8")

def models():
    st.title('Denoise your image with deep learning models..')
    
    st.write('\n')
    
    choice = st.sidebar.selectbox("Choose how to load image", ["Use Existing Images", "Browse Image"])
    
    if choice == "Browse Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image file", type="jpg")

        if uploaded_file is not None:
            # Convert the file to an OpenCV image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            gt = cv2.imdecode(file_bytes, 1)
            prediction_ui(gt)
          
    if choice == "Use Existing Images":
        image_file_chosen = st.sidebar.selectbox('Select an existing image:', get_list_of_images(), 10)
        
        if image_file_chosen:
            images_path = os.path.join(os.getcwd(), 'images')
            gt = cv2.imread(os.path.join(images_path, image_file_chosen))
            prediction_ui(gt)

def get_list_of_images():
    file_list = os.listdir(os.path.join(os.getcwd(), 'images'))
    filenames = sorted([str(filename) for filename in file_list if str(filename).endswith('.jpg')])
    return filenames
    
def prediction_ui(gt):
    models_load_state = st.text('\n Loading models..')
    dncnn, dncnn_lite, ridnet, ridnet_lite = get_models()
    models_load_state.text('\n Models Loading.. complete')

    dncnn_filesize, dncnnlite_filesize, ridnet_filesize, ridnetlite_filesize = get_filesizes()
    
    noise_level = st.sidebar.slider("Pick the noise level", 0, 45, 0)
          
    ground_truth, noisy_image, patches_noisy = get_image(gt, noise_level=noise_level)
    st.header('Input Image')
    st.markdown('** Noise level : ** `%d`  ( Noise level `0` will be same as original image )' % (noise_level))
    st.image(noisy_image)
    
    if noise_level != 0:
        st.success('PSNR of Noisy image : %.3f db' % PSNR(ground_truth, noisy_image))
      
    model = st.sidebar.radio("Choose a model to predict", ('DNCNN', 'RIDNET'), 0)
    
    submit = st.sidebar.button('Predict Now')
          
    if submit and noise_level >= 10:
        if model == 'DNCNN':
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = predict_fun(dncnn, patches_noisy, gt)
            progress_bar.progress(40)
            end = time.time()
            st.header('Denoised image using DnCNN model')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )' % (dncnn_filesize, (end - start)))
            st.image(denoised_image)     
            st.success('PSNR of denoised image : %.3f db' % (PSNR(ground_truth, denoised_image)))
            
            progress_bar.progress(60)
            start = time.time()
            denoised_image_lite = predict_fun_tflite(dncnn_lite, patches_noisy, gt)
            end = time.time()
            st.header('Denoised image using lite version of DnCNN model')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )' % (dncnnlite_filesize, (end - start)))
            progress_bar.progress(90)
            st.image(denoised_image_lite)
            st.success('PSNR of denoised image : %.3f db' % (PSNR(ground_truth, denoised_image_lite)))
            progress_bar.progress(100)
            progress_bar.empty()
        
        elif model == 'RIDNET':
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = predict_fun(ridnet, patches_noisy, gt)
            progress_bar.progress(40)
            end = time.time()
            st.header('Denoised image using Ridnet model')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )' % (ridnet_filesize, (end - start)))
            st.image(denoised_image)
            st.success('PSNR of denoised image : %.3f db' % (PSNR(ground_truth, denoised_image)))        

            progress_bar.progress(60)
            start = time.time()
            denoised_image_lite = predict_fun_tflite(ridnet_lite, patches_noisy, gt)
            end = time.time()
            st.header('Denoised image using lite version of RIDNET model')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )' % (ridnetlite_filesize, (end - start)))
            progress_bar.progress(90)
            st.image(denoised_image_lite)
            st.success('PSNR of denoised image : %.3f db' % (PSNR(ground_truth, denoised_image_lite)))
            
            progress_bar.progress(100)
            progress_bar.empty()
        
        st.write("""\n After optimization the size of the DnCNN and RIDNET models are reduced by 5 MB, 12 MB respectively and have the same performance (PSNR) as the original models.
                    But here time taken by lighter versions is more because we perform prediction on a batch of patches, thus for each patch the lite version needs
                    to invoke the model, so it's taking more than usual. """)
        st.markdown("""** Note : This application is running on CPU, speed can be further increased by using GPU ** """)         

    elif submit == True and noise_level < 10:
        st.error("Choose a minimum noise level of 10 ...")

@st.cache_resource
def get_models():
    # Define custom objects including TFOpLambda
    custom_objects = {
        'TFOpLambda': tf.keras.layers.Lambda,
        'tf': tf  # Include tf module for any tf operations
    }
    
    # Load models with custom objects
    dncnn = tf.keras.models.load_model('dncnn.h5', 
                                      compile=False,
                                      custom_objects=custom_objects)
    ridnet = tf.keras.models.load_model('ridnet.h5', 
                                       compile=False,
                                       custom_objects=custom_objects)
    
    # Load TFLite models
    dncnn_lite = tf.lite.Interpreter('dncnn2.tflite')
    ridnet_lite = tf.lite.Interpreter('ridnet.tflite')
    
    # Recompile the models
    dncnn.compile(optimizer='adam', loss='mse')
    ridnet.compile(optimizer='adam', loss='mse')
    
    return dncnn, dncnn_lite, ridnet, ridnet_lite

@st.cache_resource
def get_filesizes():
    
    dncnn_filesize = os.stat('dncnn.h5').st_size / (1024 * 1024)
    dncnnlite_filesize = os.stat('dncnn2.tflite').st_size / (1024 * 1024)
    ridnet_filesize = os.stat('ridnet.h5').st_size / (1024 * 1024)
    ridnetlite_filesize = os.stat('ridnet.tflite').st_size / (1024 * 1024)
    return dncnn_filesize, dncnnlite_filesize, ridnet_filesize, ridnetlite_filesize
        
def get_patches(image):
    '''This function creates and returns patches of the given image with a specified patch_size'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    crop_sizes = [1]
    patch_size = 40
    patches = []
    for crop_size in crop_sizes: # We will crop the image to different sizes
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h - patch_size + 1, int(patch_size / 1)):
            for j in range(0, crop_w - patch_size + 1, int(patch_size / 1)):
                x = image_scaled[i:i + patch_size, j:j + patch_size] # This gets the patch from the original image with size 40x40
                patches.append(x)
    return np.array(patches)

def get_image(gt, noise_level):
    '''This function adds noise to the image'''
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    noise = np.random.normal(0, noise_level, gt.shape)
    noisy_image = np.clip(gt + noise, 0, 255).astype(np.uint8)
    
    patches_noisy = get_patches(noisy_image)
    return gt, noisy_image, patches_noisy

def predict_fun(model, patches_noisy, gt):
    '''This function runs the prediction on the given model'''
    patches_noisy = patches_noisy.astype('float32') / 255.
    denoised_patches = model.predict(patches_noisy)
    denoised_patches = denoised_patches * 255.0
    patch_size = 40
    denoised_image = np.zeros_like(gt)

    index = 0
    for i in range(0, denoised_image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, denoised_image.shape[1] - patch_size + 1, patch_size):
            denoised_image[i:i + patch_size, j:j + patch_size] += denoised_patches[index]
            index += 1

    return denoised_image

def predict_fun_tflite(interpreter, patches_noisy, gt):
    '''This function runs the prediction using TFLite'''
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    patches_noisy = patches_noisy.astype('float32') / 255.
    denoised_patches = []
    
    for i in range(len(patches_noisy)):
        interpreter.set_tensor(input_details[0]['index'], patches_noisy[i].reshape(1, 40, 40, 3))
        interpreter.invoke()
        denoised_patch = interpreter.get_tensor(output_details[0]['index'])
        denoised_patches.append(denoised_patch)

    denoised_patches = np.array(denoised_patches).squeeze() * 255.0
    patch_size = 40
    denoised_image = np.zeros_like(gt)

    index = 0
    for i in range(0, denoised_image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, denoised_image.shape[1] - patch_size + 1, patch_size):
            denoised_image[i:i + patch_size, j:j + patch_size] += denoised_patches[index]
            index += 1

    return denoised_image

def PSNR(original, compressed):
    '''This function computes PSNR of two images'''
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

if __name__ == "__main__":
    main()
