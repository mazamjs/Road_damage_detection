import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DARK UI --------------------

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #101a2b;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

h1, h2, h3 {
    color: #f1f5f9;
}

p {
    color: #aab7ca;
}

hr {
    border-color: #29394f;
}

[data-testid="stFileUploader"] {
    background: #111c2e;
    border: 1px dashed #38516d;
    border-radius: 16px;
    padding: 18px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stMetric"] {
    background: #141f32;
    border: 1px solid #29394f;
    border-radius: 12px;
    padding: 16px;
}

.stDownloadButton button {
    background: #06b6d4;
    color: #07111f;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton button:hover {
    background: #22d3ee;
    color: #07111f;
}

.project-info {
    background: #141f32;
    border: 1px solid #29394f;
    border-radius: 12px;
    padding: 16px;
    margin-top: 16px;
}

.project-label {
    color: #22d3ee;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.project-heading {
    font-size: 17px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 8px;
}

.project-description {
    color: #aab7ca;
    font-size: 14px;
    line-height: 1.7;
}

.ready-box {
    background: #141f32;
    border: 1px solid #29394f;
    border-radius: 18px;
    padding: 65px 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")


# -------------------- SIDEBAR PROJECT DETAILS --------------------

with st.sidebar:
    st.markdown(
        """
        <div style="margin-top:35px; margin-bottom:28px;">
            <div style="font-size:25px; font-weight:800;">
                🛣️ ROADGUARD <span style="color:#22d3ee;">AI</span>
            </div>
            <div style="color:#94a3b8; font-size:12px;
                        letter-spacing:2px; margin-top:10px;">
                ROAD DAMAGE DETECTION
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="project-label">ABOUT THE PROJECT</div>

        <div class="project-heading">Road Damage Detection</div>

        <div class="project-description">
            RoadGuard AI uses a custom-trained YOLO object detection
            model to identify visible road damage in uploaded images.
        </div>

        <div class="project-heading">Key Features</div>

        <div class="project-description">
            • Automatic road damage detection<br>
            • Bounding boxes around detected damage<br>
            • Damage category identification<br>
            • Confidence scores for predictions<br>
            • Downloadable detection results
        </div>

        <div class="project-heading">Technology</div>

        <div class="project-description">
            • Python<br>
            • Streamlit<br>
            • Ultralytics YOLO<br>
            • Pillow
        </div>

        <div class="project-heading">How to Use</div>

        <div class="project-description">
            1. Upload a road image.<br>
            2. Wait for the model to inspect it.<br>
            3. Review the detected damage.<br>
            4. Download the annotated image.
        </div>

        <div class="project-info">
            <div style="color:#22d3ee; font-weight:700;">
                AI-POWERED INSPECTION
            </div>
            <div class="project-description" style="margin-top:8px;">
                Supporting faster visual road damage assessment.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------- MAIN PAGE --------------------

st.title("Image Inspection")
st.write("Upload a road image to identify damage.")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)


# -------------------- READY SCREEN --------------------

if uploaded_file is None:
    st.markdown(
        """
        <div class="ready-box">
            <div style="font-size:42px;">📷</div>
            <h2>Ready for Inspection</h2>
            <p>Upload a road image above to start detection.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------- DETECTION --------------------

else:
    if not os.path.exists("best.pt"):
        st.error("best.pt file nahi mili. Isay app.py ke saath rakho.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting road damage..."):
            model = load_model()

            result = model.predict(
                source=image,
                conf=0.50,
                verbose=False
            )[0]

        detected_image = result.plot()[:, :, ::-1]

        st.markdown("---")
        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("**Detected Road Damage**")
            st.image(detected_image, use_container_width=True)

        boxes = result.boxes
        total_detections = len(boxes) if boxes is not None else 0

        detected_classes = []

        if boxes is not None and total_detections > 0:
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].item())

                detected_classes.append({
                    "Damage Type": class_name,
                    "Confidence": f"{confidence:.2%}"
                })

        unique_classes = set(
            item["Damage Type"] for item in detected_classes
        )

        st.markdown("### Inspection Summary")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric("Total Detections", total_detections)

        with metric2:
            st.metric("Damage Classes", len(unique_classes))

        if detected_classes:
            st.markdown("### Detected Damage")

            st.dataframe(
                detected_classes,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No road damage detected in this image. "
                "Try another road image."
            )

        # Download annotated image
        output_image = Image.fromarray(detected_image)
        image_buffer = io.BytesIO()
        output_image.save(image_buffer, format="PNG")

        st.download_button(
            label="Download Detection Image",
            data=image_buffer.getvalue(),
            file_name="road_damage_detection.png",
            mime="image/png",
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Detection error: {error}")