import io
import base64
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.cm as cm

def decode_base64_image(image_data):
    """
    Decode a base64 image sent from the frontend canvas.
    """
    # Remove data URL prefix
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return image


def preprocess_image_for_model(pil_image):
    """
    Convert user-drawn image into MNIST-like format:
    - grayscale
    - resize to 28x28
    - invert colors (white digit on black background)
    - normalize to 0-1
    - reshape to (1, 28, 28, 1)
    """
    # Resize image to 28x28
    image = pil_image.resize((28, 28))

    # Invert image colors if canvas has black strokes on white background
    image = ImageOps.invert(image)

    # Convert to numpy array
    img_array = np.array(image).astype("float32") / 255.0

    # Reshape for CNN input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, image

   
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):

    # 🔹 Ensure model is built
    _ = model.predict(img_array)

    # 🔹 Find last conv layer automatically if not given
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found")

    # 🔹 Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 🔹 Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

"""def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    Generate Grad-CAM heatmap for explainability.
    
    model = tf.keras.models.Model(inputs=model.input, outputs=model.output)
    if last_conv_layer_name is None:
        # Find last Conv2D layer automatically
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()"""


def heatmap_to_base64(heatmap):
    """
    Convert heatmap array to base64 image for frontend display.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((200, 200))
    
    buffered = io.BytesIO()
    jet_heatmap.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
