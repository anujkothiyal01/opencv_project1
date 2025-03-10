import streamlit as st
import numpy as np
import cv2
import io
import base64
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Function to create a download link for output images
def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')  # Using PNG for better quality
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Sidebar title
st.sidebar.title('🖌️ Image Inpainting App')

# Upload image
uploaded_file = st.sidebar.file_uploader("📤 Upload an Image to Restore:", type=["png", "jpg", "jpeg"])
image = None
res = None

if uploaded_file:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Resize image for better performance
    h, w = image.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        h, w = int(h * scale), max_width
    else:
        h, w = h, w

    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((w, h))

    # Sidebar controls
    stroke_width = st.sidebar.slider("✏️ Stroke Width:", 1, 25, 5)

    # Create a canvas for drawing mask
    canvas_result = st_canvas(
        fill_color='white',
        stroke_width=stroke_width,
        stroke_color='black',
        background_image=pil_image,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode='freedraw',
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert mask to binary format
        mask = cv2.split(canvas_result.image_data)[3]
        mask = cv2.resize(np.uint8(mask), (image.shape[1], image.shape[0]))

        # Show mask (optional)
        if st.sidebar.checkbox('👀 Show Mask Preview'):
            st.image(mask, caption="Mask", use_column_width=True)

        # Choose inpainting mode
        st.sidebar.caption('🎯 Select Inpainting Mode:')
        option = st.sidebar.selectbox('Mode', ['None', 'Telea', 'NS', 'Compare Both'])

        if option in ['Telea', 'NS', 'Compare Both']:
            st.subheader(f"🖼️ Result of {option}")

            # Apply inpainting
            res_telea = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            res_ns = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            if option == 'Telea':
                res = res_telea[:, :, ::-1]
                st.image(res, caption="Telea Inpainting", use_column_width=True)

            elif option == 'NS':
                res = res_ns[:, :, ::-1]
                st.image(res, caption="NS Inpainting", use_column_width=True)

            elif option == 'Compare Both':
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('Telea')
                    st.image(res_telea[:, :, ::-1], use_column_width=True)
                with col2:
                    st.subheader('NS')
                    st.image(res_ns[:, :, ::-1], use_column_width=True)

            # Provide Download Links
            if res_telea is not None:
                result_telea = Image.fromarray(res_telea[:, :, ::-1])
                st.sidebar.markdown(
                    get_image_download_link(result_telea, 'telea_output.png', '📥 Download Telea Result'),
                    unsafe_allow_html=True
                )
            if res_ns is not None:
                result_ns = Image.fromarray(res_ns[:, :, ::-1])
                st.sidebar.markdown(
                    get_image_download_link(result_ns, 'ns_output.png', '📥 Download NS Result'),
                    unsafe_allow_html=True
                )
