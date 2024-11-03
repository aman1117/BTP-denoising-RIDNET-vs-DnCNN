import tensorflow as tf

# Define custom function
def custom_expand_dims(x):
    return tf.expand_dims(x, axis=-1)

# Define custom objects
custom_objects = {
    'TFOpLambda': tf.keras.layers.Lambda,
    'custom_expand_dims': custom_expand_dims
}

# Load the RIDNET model with custom objects
model = tf.keras.models.load_model('ridnet.h5', compile=False, custom_objects=custom_objects)

# Print the model summary to inspect its architecture
model.summary()