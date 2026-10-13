"""
app.py — Streamlit Ground Dashboard for SIGNALVEX
=================================================
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology
Sky Seekers — Searching the Skies. Saving Lives.

Features:
A. Video Input & Metadata inspection (or sample video selector)
B. Live Detection Display (annotated frames, bounding boxes, track IDs, timestamp banner)
C. Detection Summary (total detections, approx distinct track IDs, flagged items, measured FPS)
D. Filterable Detection Table with interactive review status (Keep / Reject / Needs Review)
E. Data Export (CSV download and annotated video download)
F. Safety, Limitations & Academic Transparency Notes
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import HumanDetector
from tracker import HumanTracker
from logger import DetectionLogger
from video_processor import VideoProcessor

# Set Streamlit page configuration
st.set_page_config(
    page_title="SIGNALVEX | Rescue Intelligence Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern SAR dashboard aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #0284c7;
        font-weight: 600;
        margin-top: 0px;
        margin-bottom: 12px;
    }
    .meta-tag {
        font-size: 0.82rem;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .callout-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.88rem;
        color: #166534;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "detection_logger" not in st.session_state:
    st.session_state.detection_logger = DetectionLogger()
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None
if "last_processed_video" not in st.session_state:
    st.session_state.last_processed_video = None


# Cached Model Loader
@st.cache_resource(show_spinner="Initializing YOLOv8n detector...")
def get_detector_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    return HumanDetector(model_path=model_path, conf_threshold=0.35)


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.markdown("### 🚁 **SIGNALVEX**")
    st.markdown("**Drone-Based Human Detection & Rescue Mapping**")
    st.caption("Chaitanya Bharathi Institute of Technology\nDept. of ECE · Sky Seekers Team")

    st.divider()
    st.markdown("#### ⚙️ Processing Parameters")

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.15,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Minimum confidence score for logging person detections. Lower picks more candidates; higher reduces false alarms."
    )

    review_threshold = st.slider(
        "Low-Confidence Flag Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Detections between Confidence Threshold and this value are highlighted in amber and flagged as 'Needs Review'."
    )

    max_frames_to_process = st.number_input(
        "Frame Limit (0 = Full Video)",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
        help="Optional limit for rapid faculty live demonstrations on CPU."
    )

    st.divider()
    st.markdown("#### 📌 Prototype Notes & Limitations")
    st.caption("""
    • **RGB Prototype:** Detects visible human silhouettes only.
    • **Life-Safety Notice:** Not a certified life-safety system; human operator in the loop is essential.
    • **Tracking Note:** Track IDs are approximate due to potential occlusions or ID switches.
    • **Scope:** No hardware or real GPS coordinates are generated at this stage.
    """)


# ---------------- MAIN HEADER ----------------
st.markdown('<div class="main-title">SIGNALVEX — Ground Rescue Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based Search-and-Rescue Support on Recorded Drone Video</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="meta-tag"><b>Architecture:</b> Recorded RGB Video ➔ Preprocess ➔ YOLO (Person Class 0) ➔ Confidence Filter ➔ ByteTrack ➔ Ground Review ➔ CSV Export</div>',
    unsafe_allow_html=True
)


# ---------------- SECTION A: VIDEO INPUT ----------------
st.subheader("1. Video Source Selection")

# Look for built-in sample videos
sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
sample_files = []
if os.path.exists(sample_dir):
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

input_method = st.radio(
    "Choose video source:",
    ["Upload Video File (.mp4, .avi, .mov)", "Use Built-in Sample Aerial Drone Video"],
    horizontal=True
)

selected_video_path = None

if input_method == "Upload Video File (.mp4, .avi, .mov)":
    uploaded_file = st.file_uploader(
        "Upload aerial footage (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        help="Select a recorded aerial or ground RGB video clip to analyze."
    )
    if uploaded_file is not None:
        # Save uploaded file to temp directory
        temp_dir = tempfile.gettempdir()
        temp_input_path = os.path.join(temp_dir, f"signalvex_upload_{uploaded_file.name}")
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())
        selected_video_path = temp_input_path
else:
    if sample_files:
        selected_sample = st.selectbox("Select packaged aerial sample:", sample_files)
        selected_video_path = os.path.join(sample_dir, selected_sample)
    else:
        st.warning("No sample files found in `sample_data/`. Please upload a video file above.")

# Video Metadata Preview
if selected_video_path and os.path.exists(selected_video_path):
    try:
        video_info = VideoProcessor.get_video_info(selected_video_path)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("File Name", video_info["file_name"])
        with col_m2:
            st.metric("Resolution", f"{video_info['width']} × {video_info['height']} px")
        with col_m3:
            st.metric("Frame Rate", f"{video_info['fps']} FPS")
        with col_m4:
            st.metric("Duration", f"{video_info['duration_formatted']} ({video_info['total_frames']} frames)")
    except Exception as e:
        st.error(f"Error inspecting video file: {e}")
        selected_video_path = None


# ---------------- PROCESSING TRIGGER ----------------
st.write("")
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    process_button = st.button("🚀 Start SAR Analysis", type="primary", disabled=(selected_video_path is None))

if process_button and selected_video_path:
    # Initialize components
    try:
        detector = get_detector_model()
        tracker = HumanTracker(model_instance=detector.model, conf_threshold=conf_thresh)
        st.session_state.detection_logger.clear()
        processor = VideoProcessor(tracker=tracker, logger=st.session_state.detection_logger)

        temp_dir = tempfile.gettempdir()
        annotated_out_path = os.path.join(temp_dir, f"signalvex_annotated_{int(time.time())}.mp4")

        # UI elements for live feedback
        st.info("Analysis in progress: scanning frames for human candidates using YOLO & ByteTrack...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        display_col1, display_col2 = st.columns([2, 1])

        with display_col1:
            live_frame_slot = st.empty()
        with display_col2:
            live_metrics_slot = st.empty()

        # Define progress callback
        def update_progress(ratio, frame, stats):
            progress_bar.progress(ratio)
            status_text.text(f"Processing Frame {stats['frame_idx']} / {stats['total_frames']} | Current Tracks: {stats['current_tracks']}")
            # Show downscaled live preview
            preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_frame_slot.image(preview_frame, caption=f"Live Feed — Frame {stats['frame_idx']}", use_container_width=True)
            live_metrics_slot.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats['current_tracks']}</div>
                <div class="metric-label">Active Targets This Frame</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{stats['total_detections']}</div>
                <div class="metric-label">Cumulative Detections</div>
            </div>
            """, unsafe_allow_html=True)

        frames_limit = int(max_frames_to_process) if max_frames_to_process > 0 else None

        # Execute analysis
        summary = processor.process_video(
            video_path=selected_video_path,
            output_path=annotated_out_path,
            conf_threshold=conf_thresh,
            max_frames=frames_limit,
            progress_callback=update_progress
        )

        st.session_state.processing_complete = True
        st.session_state.processing_summary = summary
        st.session_state.annotated_video_path = annotated_out_path
        st.session_state.last_processed_video = selected_video_path
        st.success(f"Processing successfully completed in {summary['processing_time_sec']}s at {summary['measured_fps']} FPS!")

    except Exception as e:
        st.error(f"Processing failed with error: {e}")
        st.session_state.processing_complete = False


# ---------------- SECTION B & C: RESULTS & SUMMARY ----------------
if st.session_state.processing_complete and st.session_state.processing_summary:
    summary = st.session_state.processing_summary

    st.divider()
    st.subheader("2. Search-and-Rescue Detection Summary")

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['total_detections']}</div>
            <div class="metric-label">Total Person Detections</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['approx_distinct_tracks']}</div>
            <div class="metric-label">Approx. Distinct Tracks*</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #d97706;">{summary['needs_review_count']}</div>
            <div class="metric-label">Flagged / Needs Review</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['measured_fps']}</div>
            <div class="metric-label">Measured FPS (CPU/GPU)</div>
        </div>
        """, unsafe_allow_html=True)
    with s5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['processing_time_sec']}s</div>
            <div class="metric-label">Total Execution Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("* **Note:** Distinct tracks represent tracked trajectory identifiers. Due to occlusions or camera angles, this is an approximate count rather than guaranteed unique survivors.")

    # ---------------- SECTION D & E: DETECTION TABLE & HUMAN REVIEW ----------------
    st.divider()
    st.subheader("3. Human Operator Review & Detection Log")

    df = st.session_state.detection_logger.to_dataframe()

    if not df.empty:
        # Filtering controls
        col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
        with col_f1:
            status_filter = st.multiselect(
                "Filter by Review Status:",
                options=["Keep", "Reject", "Needs Review"],
                default=["Keep", "Reject", "Needs Review"]
            )
        with col_f2:
            min_conf_filter = st.slider(
                "Filter Minimum Confidence:",
                min_value=0.0,
                max_value=1.0,
                value=float(conf_thresh),
                step=0.05
            )
        with col_f3:
            track_ids = sorted(list(df["track_id"].unique()))
            selected_track = st.selectbox(
                "Filter by Track ID:",
                options=["All Tracks"] + [f"Track #{t}" for t in track_ids]
            )

        # Apply filtering
        filtered_df = df[
            (df["review_status"].isin(status_filter)) &
            (df["confidence"] >= min_conf_filter)
        ]
        if selected_track != "All Tracks":
            tid = int(selected_track.split("#")[1])
            filtered_df = filtered_df[filtered_df["track_id"] == tid]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** logged detections.")

        # Display table with clean styling
        display_cols = [
            "frame_number", "timestamp_seconds", "track_id",
            "confidence_percent", "bbox_str", "review_status"
        ]
        st.dataframe(
            filtered_df[display_cols].rename(columns={
                "frame_number": "Frame #",
                "timestamp_seconds": "Time (s)",
                "track_id": "Track ID",
                "confidence_percent": "Confidence (%)",
                "bbox_str": "Bounding Box [x1, y1, x2, y2]",
                "review_status": "Operator Status"
            }),
            use_container_width=True,
            height=280
        )

        # Human Review Action Form
        st.markdown("#### ✍️ Update Operator Review Status")
        col_act1, col_act2, col_act3 = st.columns([2, 2, 2])

        with col_act1:
            target_track = st.selectbox(
                "Select Track to Update:",
                options=track_ids,
                format_func=lambda x: f"Track #{x}"
            )
        with col_act2:
            new_decision = st.selectbox(
                "Assign Decision:",
                ["Keep", "Reject", "Needs Review"]
            )
        with col_act3:
            st.write("")
            st.write("")
            if st.button("Apply Decision to Track"):
                st.session_state.detection_logger.update_status_by_track(target_track, new_decision)
                st.success(f"Updated all records for Track #{target_track} to '{new_decision}'!")
                st.rerun()

        # Export & Download Section
        st.divider()
        st.subheader("4. Data Export & Verification")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            csv_data = st.session_state.detection_logger.to_csv()
            st.download_button(
                label="📥 Download Detection Log (CSV)",
                data=csv_data,
                file_name="signalvex_detection_log.csv",
                mime="text/csv",
                help="Exports frame numbers, timestamps, tracking IDs, confidence scores, coordinates, and operator review decisions."
            )

        with col_exp2:
            annotated_path = st.session_state.annotated_video_path
            if annotated_path and os.path.exists(annotated_path):
                with open(annotated_path, "rb") as vf:
                    st.download_button(
                        label="📥 Download Annotated Video (MP4)",
                        data=vf.read(),
                        file_name="signalvex_annotated_video.mp4",
                        mime="video/mp4",
                        help="Download the processed video with bounding boxes and tracking IDs burned in."
                    )
            else:
                st.info("Annotated video file is unavailable or was skipped.")

    else:
        st.warning("No human detections were recorded above the confidence threshold.")


# ---------------- FOOTER: LIMITATIONS AND ACADEMIC COMPLIANCE ----------------
st.divider()
with st.expander("ℹ️ Academic Compliance, Limitations & Next Stages (From Presentation)", expanded=False):
    st.markdown("""
    ### SIGNALVEX Search & Rescue Scope Verification

    1. **Visual Model Scope:** Pretrained YOLOv8n on visible RGB spectrum detects human silhouettes. It **cannot** diagnose injury, confirm vital signs, or penetrate rubble/obstructions.
    2. **Tracking ID Consistency:** ByteTrack associates detections across consecutive frames. ID switches can occur during prolonged occlusions or abrupt drone motion.
    3. **Localization Clarification:** In accordance with the project guidelines, **no GPS or real coordinates are invented or claimed** in this submission. Pixel coordinates `[x1, y1, x2, y2]` represent image space. Simulated flat-ground geo-projection will be connected in future research once calibrated sensor telemetry (altitude, pitch, roll) is available.
    4. **Safety Disclaimer:** This software prototype is an academic decision-support tool and is **not** a certified life-safety product.
    """)
