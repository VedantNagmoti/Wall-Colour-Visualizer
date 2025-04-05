import os
import tarfile
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# -------------------------------
# Constants
# -------------------------------
MODEL_URL = 'http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
MODEL_TARBALL_PATH = 'deeplabv3_model.tar.gz'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

# -------------------------------
# Download Model If Needed
# -------------------------------
def download_model_if_needed():
    if not os.path.exists(MODEL_TARBALL_PATH):
        with st.spinner("Downloading DeepLabV3 model (~400MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_TARBALL_PATH)
            st.success("Model downloaded!")

# -------------------------------
# DeepLab Model Loader
# -------------------------------
@st.cache_resource
def load_model(tarball_path):
    class DeepLabModel:
        INPUT_TENSOR_NAME = 'ImageTensor:0'
        OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

        def __init__(self, tarball_path):
            self.graph = tf.Graph()
            graph_def = None
            with tarfile.open(tarball_path) as tar:
                for tar_info in tar.getmembers():
                    if FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                        file_handle = tar.extractfile(tar_info)
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(file_handle.read())
                        break
            if graph_def is None:
                raise RuntimeError('Cannot find inference graph in tar archive.')

            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph)

        def run(self, image):
            width, height = image.size
            resize_ratio = 513.0 / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
            seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]
            })[0]
            return resized_image, seg_map

    return DeepLabModel(tarball_path)

# -------------------------------
# Wall Recoloring Function
# -------------------------------
def apply_wall_style(original_image, seg_map, color=None, texture_img=None, opacity=128):
    original_image = original_image.convert("RGBA")
    wall_mask = (seg_map == 1).astype(np.uint8) * 255
    wall_mask_image = Image.fromarray(wall_mask).resize(original_image.size)

    if texture_img:
        texture = texture_img.resize(original_image.size).convert("RGBA")
        texture.putalpha(wall_mask_image)
        styled_wall = texture
    else:
        color = color or (255, 0, 0)
        styled_wall = Image.new("RGBA", original_image.size, color + (0,))
        styled_wall.putalpha(wall_mask_image.point(lambda p: opacity if p > 0 else 0))

    result = Image.alpha_composite(original_image, styled_wall)
    return result

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Wall Color Styler", layout="wide")
st.title("🎨 Wall Color Styler using DeepLabV3")

# Ensure model is downloaded
download_model_if_needed()

uploaded_image = st.file_uploader("Upload an interior image", type=['jpg', 'jpeg', 'png'])

# Color/texture options
use_texture = st.checkbox("Use texture instead of color")

texture_img = None
color = None
opacity = st.slider("Wall Style Opacity (for realism)", 0, 255, 128)

if use_texture:
    texture_file = st.file_uploader("Upload texture image", type=['jpg', 'jpeg', 'png'])
    if texture_file:
        texture_img = Image.open(texture_file)
else:
    st.markdown("**Pick a wall color (20 presets available):**")
    preset_colors = [
        "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
        "#00FFFF", "#FF00FF", "#C0C0C0", "#808080", "#800000",
        "#808000", "#008000", "#800080", "#008080", "#000080",
        "#FFA500", "#A52A2A", "#F0E68C", "#ADD8E6", "#90EE90"
    ]
    color_hex = st.selectbox("Choose a preset color", preset_colors, index=5)
    color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

if uploaded_image and (color or texture_img):
    with st.spinner("Processing image..."):
        model = load_model(MODEL_TARBALL_PATH)
        input_img = Image.open(uploaded_image)
        resized_img, seg_map = model.run(input_img)
        output_img = apply_wall_style(resized_img, seg_map, color=color, texture_img=texture_img, opacity=opacity)

    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(resized_img, caption="Original Resized", use_column_width=True)

    with col2:
        st.image(output_img, caption="Styled Output", use_column_width=True)
