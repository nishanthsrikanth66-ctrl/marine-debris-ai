import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd

from preprocessing import preprocess
from detector import detect
from database import create_database, save_detection


# ==========================================
# Initialize Database
# ==========================================

create_database()


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Marine Debris AI",
    page_icon="🌊",
    layout="wide"
)


# ==========================================
# Session State
# ==========================================

if "saved_detections" not in st.session_state:
    st.session_state.saved_detections = set()


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("🌊 Marine Debris AI")

    st.write("### System")

    st.success("🟢 AI System Online")

    st.write("---")

    st.write("### AI Pipeline")

    st.write("📡 Side-Scan Sonar")
    st.write("🔧 OpenCV Preprocessing")
    st.write("🤖 YOLOv8 Detection")
    st.write("💾 SQLite Database")
    st.write("📊 Detection Reports")

    st.write("---")

    st.info(
        "Upload a sonar image to detect "
        "underwater objects and anomalies."
    )


# ==========================================
# Main Title
# ==========================================

st.title("🌊 Marine Debris AI Detection System")

st.write(
    "AI-powered detection and classification of "
    "underwater marine debris and suspicious sonar anomalies."
)


# ==========================================
# Dashboard Statistics
# ==========================================

conn = sqlite3.connect("database/detections.db")

stats = pd.read_sql_query(
    """
    SELECT
        COUNT(*) AS total_detections,
        COUNT(DISTINCT target_class) AS object_types,
        AVG(confidence) AS average_confidence
    FROM detections
    """,
    conn
)

conn.close()


total_detections = int(
    stats["total_detections"][0]
)

object_types = int(
    stats["object_types"][0]
)

average_confidence = (
    float(stats["average_confidence"][0])
    if stats["average_confidence"][0] is not None
    else 0
)


# ==========================================
# Statistics
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🎯 Total Detections",
        total_detections
    )

with col2:
    st.metric(
        "🔎 Object Types",
        object_types
    )

with col3:
    st.metric(
        "📈 Average Confidence",
        f"{average_confidence:.2%}"
    )


st.write("---")


# ==========================================
# Image Upload
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload a side-scan sonar image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# Process Uploaded Image
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # Read Image
    # --------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------
    # Convert PIL → OpenCV
    # --------------------------------------

    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


    # ======================================
    # OpenCV Preprocessing
    # ======================================

    st.subheader("🔧 Sonar Image Preprocessing")

    processed_image = preprocess(
        image_cv
    )


    col1, col2 = st.columns(2)

    with col1:

        st.write("### 📷 Original Sonar Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.write("### 🔧 Processed Sonar Image")

        st.image(
            processed_image,
            caption="Noise Reduced + Contrast Enhanced",
            use_container_width=True
        )


    # ======================================
    # YOLO Detection
    # ======================================

    st.subheader("🤖 AI Detection")

    with st.spinner(
        "AI model is analyzing the sonar image..."
    ):

        detections = detect(
            image_cv
        )


    # ======================================
    # Detection Results
    # ======================================

    if detections:

        result_image = image_cv.copy()


        # ----------------------------------
        # Draw Detection Boxes
        # ----------------------------------

        for detection in detections:

            x1 = int(
                detection["x1"]
            )

            y1 = int(
                detection["y1"]
            )

            x2 = int(
                detection["x2"]
            )

            y2 = int(
                detection["y2"]
            )


            confidence = detection[
                "confidence"
            ]

            class_name = detection[
                "class_name"
            ]


            # ------------------------------
            # Bounding Box
            # ------------------------------

            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------
            # Detection Label
            # ------------------------------

            label = (
                f"{class_name}: "
                f"{confidence:.2f}"
            )


            cv2.putText(
                result_image,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================
            # Unique Detection Key
            # ==================================

            detection_key = (
                uploaded_file.name,
                class_name,
                round(confidence, 5),
                x1,
                y1,
                x2,
                y2
            )


            # ==================================
            # Save Detection Only Once
            # ==================================

            if (
                detection_key
                not in
                st.session_state.saved_detections
            ):

                save_detection(
                    image_name=uploaded_file.name,
                    target_class=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )

                st.session_state.saved_detections.add(
                    detection_key
                )


        # ==================================
        # Convert BGR → RGB
        # ==================================

        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )


        # ==================================
        # Display Results
        # ==================================

        st.subheader(
            "🎯 Detection Results"
        )

        st.image(
            result_image,
            caption="YOLOv8 Detection Results",
            use_container_width=True
        )


        # ==================================
        # Detection Count
        # ==================================

        st.success(
            f"✅ Detected {len(detections)} object(s)"
        )


        # ==================================
        # Detection Details
        # ==================================

        st.subheader(
            "📊 Detection Details"
        )


        for i, detection in enumerate(
            detections,
            start=1
        ):

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]


            st.write(
                f"**{i}. {class_name}** — "
                f"Confidence: "
                f"**{confidence:.2%}**"
            )


    else:

        st.info(
            "ℹ️ No objects detected."
        )


    # ======================================
    # Completion Message
    # ======================================

    st.success(
        "✅ Image analysis completed successfully!"
    )


# ==========================================
# Detection History
# ==========================================

st.write("---")

st.subheader(
    "📜 Detection History"
)


conn = sqlite3.connect(
    "database/detections.db"
)


history = pd.read_sql_query(
    """
    SELECT
        id,
        image_name,
        target_class,
        confidence,
        timestamp
    FROM detections
    ORDER BY id DESC
    """,
    conn
)


conn.close()


# ==========================================
# Display History
# ==========================================

if not history.empty:

    history["confidence"] = (
        history["confidence"] * 100
    ).round(2)


    st.dataframe(
        history,
        use_container_width=True
    )


    # ======================================
    # CSV Report
    # ======================================

    csv = history.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name=(
            "marine_debris_detection_report.csv"
        ),
        mime="text/csv"
    )


else:

    st.info(
        "No detection history available yet."
    )


# ==========================================
# Footer
# ==========================================

st.write("---")

st.caption(
    "🌊 Marine Debris AI | "
    "OpenCV + YOLOv8 + Streamlit + SQLite"
)