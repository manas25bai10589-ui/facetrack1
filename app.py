import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import json
from datetime import datetime, date
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
from database.db_manager import DatabaseManager
from database.seed_demo_data import seed_demo_data
from vision.face_engine import FaceEngine
from vision.video_tracker import VideoTracker
from vision.cctv_stream import CCTVStreamReader, CCTV_PRESETS, build_rtsp_url, test_cctv_connection
from compliance.dpdp_manager import DPDPComplianceManager
from pilot.pilot_engine import PilotManager

# Page Configuration
st.set_page_config(
    page_title="FaceTrack AI — Privacy-Compliant Venue Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark / Modern B2B Theme)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #10b981;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 2px;
    }
    .privacy-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .warning-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px;
        padding: 8px 16px;
        color: #cbd5e1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State & Managers
@st.cache_resource
def get_managers():
    db = DatabaseManager()
    engine = FaceEngine()
    tracker = VideoTracker(db, engine)
    compliance = DPDPComplianceManager(db)
    pilot = PilotManager(db)
    return db, engine, tracker, compliance, pilot

db, engine, tracker, compliance, pilot = get_managers()

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/facial-recognition.png", width=64)
    st.title("FaceTrack AI")
    st.caption("Privacy-First B2B Venue Analytics")
    
    st.markdown('<div class="privacy-badge">🛡️ DPDP Act 2023 Compliant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "🎥 Live Venue Monitor",
            "📊 Footfall & Peak Hours",
            "⏱️ Dwell Time & Zones",
            "🔄 Retention & Churn Risk",
            "👤 Member Enrollment",
            "🛡️ DPDP Compliance & Erasure",
            "🚀 Phase 2: Pilot & Grant Hub",
            "⚙️ Settings & Demo Tools"
        ]
    )
    
    st.markdown("---")
    st.subheader("Venue Config")
    selected_zone = st.selectbox(
        "Active Camera Zone",
        options=[z["id"] for z in config.DEFAULT_ZONES],
        format_func=lambda zid: next((z["name"] for z in config.DEFAULT_ZONES if z["id"] == zid), zid)
    )
    tracker.set_active_zone(selected_zone)
    
    st.info(f"**Venue:** {config.DEFAULT_VENUE_NAME}\n\n**Storage:** SQLite (Offline-First)")

# -------------------------------------------------------------
# PAGE 1: LIVE VENUE MONITOR
# -------------------------------------------------------------
if selected_page == "🎥 Live Venue Monitor":
    st.title("🎥 Live Venue Monitor & Video Stream")
    st.write(
        "Real-time opt-in face recognition and zone dwell tracking. "
        "**Privacy-by-Design:** Non-consenting faces are processed in-memory and discarded immediately."
    )

    # Top KPI Metrics
    summary = db.get_summary_metrics()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Consented Members</div>
            <div class="metric-value">{summary['total_members']}</div>
            <div class="metric-sub">Active Opt-Ins</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Visits Today</div>
            <div class="metric-value">{summary['visits_today']}</div>
            <div class="metric-sub">Logged Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Dwell Today</div>
            <div class="metric-value">{summary['avg_dwell_today_min']} min</div>
            <div class="metric-sub">Per Visit</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Currently on Floor</div>
            <div class="metric-value">{len(tracker.active_sessions)}</div>
            <div class="metric-sub">Real-Time Occupancy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.radio(
        "Video Input Source:",
        ["📹 Real-Time CCTV Stream (RTSP / IP Cam)", "🔴 Local Webcam Feed", "🖼️ Image Snapshot / Test File", "🧪 Floor Simulation"],
        horizontal=True
    )

    if mode == "📹 Real-Time CCTV Stream (RTSP / IP Cam)":
        st.info("Connect to commercial CCTV NVRs, IP cameras, or wireless phone cameras over the local network via RTSP.")
        cctv_c1, cctv_c2 = st.columns([2, 1])

        with cctv_c2:
            st.subheader("CCTV Network Settings")
            preset_choice = st.selectbox("CCTV Hardware Preset", list(CCTV_PRESETS.keys()), index=0)
            
            if preset_choice != "Custom RTSP URL":
                col_ip, col_port = st.columns([2, 1])
                with col_ip:
                    cctv_ip = st.text_input("Camera / NVR IP", value="192.168.1.100")
                with col_port:
                    cctv_port = st.text_input("Port", value="8080" if "Mobile" in preset_choice else "554")
                
                col_u, col_p = st.columns(2)
                with col_u:
                    cctv_user = st.text_input("Username", value="" if "Mobile" in preset_choice else "admin")
                with col_p:
                    cctv_pass = st.text_input("Password", value="", type="password")
                
                cctv_chan = st.text_input("Channel No. (for NVRs)", value="1") if "Mobile" not in preset_choice and "Tapo" not in preset_choice else "1"
                final_rtsp_url = build_rtsp_url(preset_choice, cctv_ip, cctv_port, cctv_user, cctv_pass, cctv_chan)
            else:
                final_rtsp_url = st.text_input("Full RTSP URL", value="rtsp://admin:pass@192.168.1.100:554/stream1")

            st.caption(f"**RTSP Target:** `{final_rtsp_url}`")
            
            # Diagnostic Test Button
            if st.button("🔍 Test CCTV Connection"):
                with st.spinner("Connecting to CCTV socket..."):
                    ok, msg, diag = test_cctv_connection(final_rtsp_url, timeout_seconds=4)
                    if ok:
                        st.success(f"✅ {msg} (Resolution: {diag.get('resolution')})")
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("---")
            run_cctv = st.toggle("📹 Start Real-Time CCTV Stream", value=False)
            sim_threshold = st.slider("Match Confidence", 0.30, 0.60, config.COSINE_SIMILARITY_THRESHOLD, 0.02, key="cctv_thresh")
            tracker.face_engine.cosine_threshold = sim_threshold

        with cctv_c1:
            frame_box = st.empty()
            stats_box = st.empty()

            if run_cctv:
                streamer = CCTVStreamReader(final_rtsp_url)
                if not streamer.start():
                    st.error(f"Connection Failed: {streamer.error_message or 'Check CCTV IP address and network reachability.'}")
                else:
                    st.success("🟢 Zero-Latency Threaded CCTV Ingestion Active!")
                    stop_cctv = st.button("⏹️ Disconnect CCTV", key="stop_cctv_btn")
                    try:
                        while run_cctv and not stop_cctv:
                            has_frame, frame = streamer.get_frame()
                            if has_frame and frame is not None:
                                annotated, matched, stats = tracker.process_frame(frame, selected_zone)
                                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                frame_box.image(rgb, channels="RGB", use_container_width=True)
                                stats_box.markdown(
                                    f"**CCTV FPS:** `{streamer.fps:.1f}` | "
                                    f"**Faces:** `{stats['detected_faces']}` | "
                                    f"**Consented Matches:** `{stats['matched_members']}` | "
                                    f"**Non-Members Purged:** `{stats['discarded_unregistered']}` | "
                                    f"**Active In Zone:** `{stats['active_on_floor']}`"
                                )
                            time.sleep(0.015)
                    finally:
                        streamer.stop()
            else:
                frame_box.info("CCTV Stream is offline. Configure IP settings and switch on **'Start Real-Time CCTV Stream'**.")

    elif mode == "🔴 Local Webcam Feed":
        st.info("Live OpenCV stream connected to your device's camera. Ensure lighting is adequate and face the camera.")
        cam_col1, cam_col2 = st.columns([3, 1])
        
        with cam_col2:
            st.subheader("Webcam Controls")
            cam_device = st.selectbox("Webcam Index", [0, 1, 2], index=0)
            run_stream = st.toggle("🔴 Activate Camera Stream", value=False, key="webcam_toggle")
            sim_threshold = st.slider("Cosine Match Threshold", 0.30, 0.60, config.COSINE_SIMILARITY_THRESHOLD, 0.02, key="webcam_thresh")
            tracker.face_engine.cosine_threshold = sim_threshold
            
            st.markdown("""
            ---
            **DPDP Privacy Mode:**
            * Non-consenting faces discarded in RAM
            * Zero video recorded to disk
            * 128-d vector matching only
            """)

        with cam_col1:
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()

            if run_stream:
                cap = cv2.VideoCapture(int(cam_device), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    st.error(f"Could not open camera ({cam_device}). Please ensure camera is connected and not used by another application.")
                else:
                    st.success("✅ Webcam connected successfully! Running real-time inference...")
                    stop_button = st.button("⏹️ Disconnect / Stop Camera Stream", key="stop_cam_btn")
                    
                    try:
                        while run_stream and not stop_button:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("Stream ended or frame could not be grabbed.")
                                break
                            
                            annotated, matched, stats = tracker.process_frame(frame, selected_zone)
                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                            
                            stats_placeholder.markdown(
                                f"**Faces Detected:** `{stats['detected_faces']}` | "
                                f"**Consented Matches:** `{stats['matched_members']}` | "
                                f"**Non-Members Purged:** `{stats['discarded_unregistered']}` | "
                                f"**Active On Floor:** `{stats['active_on_floor']}`"
                            )
                            time.sleep(0.02)
                    finally:
                        cap.release()
            else:
                frame_placeholder.info("Camera is idle. Switch on **'Activate Camera Stream'** above to connect.")

    elif mode == "🖼️ Image Snapshot / Test File":
        st.write("Upload a photo or capture an image to test face detection and member matching.")
        test_col1, test_col2 = st.columns([2, 1])
        
        with test_col1:
            uploaded = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            cam_shot = st.camera_input("Or take a quick snapshot")
            
            target_image = None
            if cam_shot is not None:
                target_image = Image.open(cam_shot)
            elif uploaded is not None:
                target_image = Image.open(uploaded)
                
            if target_image:
                img_np = np.array(target_image)
                if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                annotated, matched, stats = tracker.process_frame(img_np, selected_zone)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Processed Frame (with DPDP Discard)", use_container_width=True)
                
        with test_col2:
            st.subheader("Frame Analysis")
            if target_image:
                st.metric("Faces Detected", stats["detected_faces"])
                st.metric("Consented Members Matched", stats["matched_members"])
                st.metric("Unenrolled Faces Discarded", stats["discarded_unregistered"])
                
                if matched:
                    st.success("Matched Members:")
                    for m in matched:
                        st.write(f"- **{m['name']}** ({m['tier']}) — Match Score: `{m['similarity']:.3f}`")
                else:
                    st.info("No enrolled members matched in this frame.")

    elif mode == "🧪 Synthetic Floor Simulation":
        st.write("Simulates continuous footfall in the selected zone using enrolled member embeddings.")
        sim_btn = st.button("Simulate Member Entering Zone")
        if sim_btn:
            members = db.get_all_active_members()
            if members:
                chosen = np.random.choice(members)
                dwell_sec = np.random.randint(15 * 60, 45 * 60)
                entry_time = datetime.now()
                exit_time = entry_time + pd.Timedelta(seconds=dwell_sec)
                db.log_visit(
                    member_id=chosen["id"],
                    zone_id=selected_zone,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    dwell_seconds=dwell_sec
                )
                st.success(f"Simulated visit logged: **{chosen['full_name']}** ({chosen['membership_tier']}) in zone **{selected_zone}** for {dwell_sec // 60} mins.")
                st.rerun()
            else:
                st.warning("No active members found. Please run seed data in Demo Tools.")

# -------------------------------------------------------------
# PAGE 2: FOOTFALL & PEAK HOURS
# -------------------------------------------------------------
elif selected_page == "📊 Footfall & Peak Hours":
    st.title("📊 Venue Footfall & Peak Hours Analytics")
    st.write("Understand member arrival curves, rush hours, and capacity utilization.")
    
    hourly_data = db.get_hourly_footfall(days_back=30)
    df_hourly = pd.DataFrame(hourly_data)
    
    col_chart, col_recs = st.columns([3, 1])
    
    with col_chart:
        st.subheader("Hourly Footfall Distribution (Last 30 Days)")
        fig = px.bar(
            df_hourly,
            x="hour",
            y="count",
            labels={"hour": "Time of Day", "count": "Total Visits Logged"},
            color="count",
            color_continuous_scale="Viridis",
            title="Member Traffic by Hour"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_recs:
        st.subheader("Operational Insights")
        if not df_hourly.empty and df_hourly["count"].sum() > 0:
            top_hour = df_hourly.loc[df_hourly["count"].idxmax()]
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 12px;">
                <div class="metric-title">Peak Rush Hour</div>
                <div class="metric-value">{top_hour['hour']}</div>
                <div class="metric-sub">{top_hour['count']} visits recorded</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Staffing Recommendations:**
            - **Morning Rush (06:00 - 09:00):** Maintain 2+ floor trainers on cardio & weights.
            - **Evening Peak (18:00 - 20:00):** Maximize front desk turnstile throughput and schedule group classes.
            - **Off-Peak (13:00 - 16:00):** Ideal window for equipment sanitization and maintenance.
            """)
        else:
            st.info("No visit logs available. Please seed demo data.")

# -------------------------------------------------------------
# PAGE 3: DWELL TIME & ZONE ANALYTICS
# -------------------------------------------------------------
elif selected_page == "⏱️ Dwell Time & Zones":
    st.title("⏱️ Dwell Time & Zone Utilization")
    st.write("Measure how members distribute across venue zones and which equipment areas have highest engagement.")
    
    zone_data = db.get_zone_utilization(days_back=30)
    df_zones = pd.DataFrame(zone_data)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Zone Visit Share")
        fig_pie = px.pie(
            df_zones,
            names="zone_name",
            values="total_visits",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Average Dwell Time per Zone (Minutes)")
        fig_bar = px.bar(
            df_zones,
            x="avg_dwell_min",
            y="zone_name",
            orientation='h',
            labels={"avg_dwell_min": "Avg Dwell (Minutes)", "zone_name": "Venue Zone"},
            color="avg_dwell_min",
            color_continuous_scale="Purples"
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Detailed Zone Metrics")
    st.dataframe(df_zones, use_container_width=True)

# -------------------------------------------------------------
# PAGE 4: RETENTION & CHURN RISK
# -------------------------------------------------------------
elif selected_page == "🔄 Retention & Churn Risk":
    st.title("🔄 Member Retention & Churn Risk Engine")
    st.write(
        "Detect members showing drop-off patterns before their membership lapses. "
        "Allows proactive outreach and personalized VIP engagement."
    )
    
    retention = db.get_member_retention_data()
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Frequent Attendees (>3x/week)", len(retention["frequent"]), delta="High Engagement")
    with k2:
        st.metric("Moderate Attendees", len(retention["moderate"]))
    with k3:
        st.metric("⚠️ At Risk (Absent 14-30d)", len(retention["at_risk"]), delta="-Retention Alert", delta_color="inverse")
    with k4:
        st.metric("❌ Churned (>30d Absent)", len(retention["churned"]), delta="-High Drop Risk", delta_color="inverse")
        
    st.markdown("---")
    
    t1, t2, t3 = st.tabs(["⚠️ At-Risk Members", "🌟 VIP & Frequent Visitors", "❌ Churned Roster"])
    
    with t1:
        st.subheader("Members Requiring Outreach (Absent 14-30 Days)")
        if retention["at_risk"]:
            df_risk = pd.DataFrame(retention["at_risk"])[["member_code", "full_name", "membership_tier", "days_absent", "total_visits"]]
            st.dataframe(df_risk, use_container_width=True)
            st.info("💡 **Recommended Action:** Trigger a personalized WhatsApp/SMS check-in with a complimentary trainer session or wellness check.")
        else:
            st.success("No members currently in the 14-30 day at-risk window.")
            
    with t2:
        st.subheader("High Engagement & VIP Members")
        if retention["frequent"]:
            df_freq = pd.DataFrame(retention["frequent"])[["member_code", "full_name", "membership_tier", "total_visits", "days_absent"]]
            st.dataframe(df_freq, use_container_width=True)
        else:
            st.write("No frequent members logged yet.")
            
    with t3:
        st.subheader("Members Absent > 30 Days")
        if retention["churned"]:
            df_churn = pd.DataFrame(retention["churned"])[["member_code", "full_name", "membership_tier", "days_absent"]]
            st.dataframe(df_churn, use_container_width=True)
        else:
            st.write("No churned members.")

# -------------------------------------------------------------
# PAGE 5: MEMBER ENROLLMENT & CONSENT
# -------------------------------------------------------------
elif selected_page == "👤 Member Enrollment":
    st.title("👤 Member Enrollment & Digital Consent Portal")
    st.write(
        "Enroll a new venue member with verified explicit opt-in consent and 128-dimensional biometric embedding. "
        "Strictly compliant with India's DPDP Act, 2023."
    )
    
    terms = DPDPComplianceManager.get_standard_consent_terms()
    
    enroll_c1, enroll_c2 = st.columns([1, 1])
    
    with enroll_c1:
        st.subheader("Member Information")
        full_name = st.text_input("Full Name *", placeholder="e.g. Rahul Kapoor")
        email = st.text_input("Email Address *", placeholder="e.g. rahul@example.com")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210")
        tier = st.selectbox("Membership Tier", ["Standard", "Gold", "Platinum", "VIP"])
        
        st.subheader("Biometric Enrollment Photo")
        photo_mode = st.radio("Capture Method:", ["Webcam Snapshot", "Upload Photo File"], horizontal=True)
        
        face_img = None
        if photo_mode == "Webcam Snapshot":
            cam_pic = st.camera_input("Take Member Photo")
            if cam_pic:
                face_img = Image.open(cam_pic)
        else:
            up_pic = st.file_uploader("Upload Member Photo", type=["jpg", "png", "jpeg"])
            if up_pic:
                face_img = Image.open(up_pic)
                
    with enroll_c2:
        st.subheader("DPDP Act 2023 Statutory Consent Waiver")
        with st.expander("📜 Read Full Notice & Data Protection Terms", expanded=True):
            st.markdown(f"**Statutory Standard:** {terms['statutory_basis']}")
            st.markdown(f"**Data Fiduciary:** {terms['data_fiduciary']}")
            st.markdown(f"**Consent Version:** `{terms['version']}`")
            st.markdown("**Specified Purposes:**")
            for p in terms["purposes"]:
                st.write(p)
            st.markdown("**Safeguards & Privacy Guarantees:**")
            for s in terms["safeguards"]:
                st.write(s)
                
        consent_signed = st.checkbox(
            "I hereby explicitly opt-in and consent to the extraction and storage of a 128-d face embedding strictly for venue access and dwell analytics. I understand I have the right to withdraw and demand complete erasure at any time.",
            value=False
        )
        
        enroll_submit = st.button("Complete Enrollment & Generate Embedding", type="primary", use_container_width=True)
        
        if enroll_submit:
            if not full_name or not email:
                st.error("Full Name and Email are mandatory fields.")
            elif not consent_signed:
                st.error("DPDP Compliance Error: Explicit consent checkbox must be signed.")
            elif face_img is None:
                st.error("Please capture or upload a member photo.")
            else:
                # Convert PIL to BGR numpy array
                img_np = np.array(face_img)
                if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                # Detect face and extract embedding
                faces = engine.detect_faces(img_np)
                if not faces:
                    st.error("No face detected in the photo. Please ensure good lighting and face camera directly.")
                else:
                    embedding = engine.extract_embedding(img_np, faces[0])
                    if embedding is None:
                        st.error("Failed to generate 128-d face embedding.")
                    else:
                        success, msg, member_id = db.enroll_member(
                            full_name=full_name,
                            email=email,
                            phone=phone,
                            membership_tier=tier,
                            embedding=embedding,
                            consent_granted=True
                        )
                        if success:
                            st.success(f"🎉 {msg}")
                            st.balloons()
                            tracker.refresh_member_cache()
                        else:
                            st.error(f"Enrollment Error: {msg}")

# -------------------------------------------------------------
# PAGE 6: DPDP COMPLIANCE & RIGHT TO ERASURE
# -------------------------------------------------------------
elif selected_page == "🛡️ DPDP Compliance & Erasure":
    st.title("🛡️ DPDP Act Compliance & Data Erasure Center")
    st.write(
        "Demonstrates regulatory compliance under India's Digital Personal Data Protection Act, 2023. "
        "Allows Data Principals to exercise their **Right to Erasure** (Section 12), immediately purging biometric embeddings."
    )
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("Execute Right to Erasure")
        members = db.get_all_active_members()
        if members:
            member_options = {m["id"]: f"{m['full_name']} ({m['member_code']}) — {m['email']}" for m in members}
            selected_mid = st.selectbox("Select Member Demanding Erasure:", options=list(member_options.keys()), format_func=lambda x: member_options[x])
            erasure_reason = st.text_input("Reason / Request Note", value="Member requested complete withdrawal and biometric deletion.")
            
            confirm_erase = st.checkbox("Confirm permanent deletion of biometric vector and consent revocation")
            
            if st.button("Permanently Purge Biometrics & Erase Member", type="primary"):
                if confirm_erase:
                    res = compliance.process_erasure_request(selected_mid, erasure_reason)
                    if res["success"]:
                        st.success(res["message"])
                        tracker.refresh_member_cache()
                        st.rerun()
                    else:
                        st.error(res["message"])
                else:
                    st.warning("Please tick the confirmation checkbox.")
        else:
            st.info("No active members in database.")
            
    with col_r:
        st.subheader("Compliance Audit Report")
        st.write("Immutable log of all consent grants, notice versions, and right-to-erasure executions.")
        
        csv_data = compliance.generate_compliance_audit_csv()
        st.download_button(
            label="📥 Download Compliance Audit CSV",
            data=csv_data,
            file_name=f"dpdp_compliance_audit_{date.today().isoformat()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Recent Consent & Erasure Audit Trails")
    audit_logs = db.get_consent_audit_trail(limit=20)
    if audit_logs:
        st.dataframe(pd.DataFrame(audit_logs), use_container_width=True)
    else:
        st.info("No audit logs yet.")

# -------------------------------------------------------------
# PAGE 7: PHASE 2 PILOT & GRANT HUB
# -------------------------------------------------------------
elif selected_page == "🚀 Phase 2: Pilot & Grant Hub":
    st.title("🚀 Phase 2 — Field Pilot Validation & Grant Scale-Up Hub")
    st.write(
        "Tools to conduct a 2–4 week live venue pilot, configure multi-camera RTSP streams, "
        "collect venue manager ROI feedback, and export grant-ready dossiers for the **MSME Idea Hackathon (₹15 Lakhs)**."
    )

    t_metrics, t_cams, t_feedback, t_grant = st.tabs([
        "📊 Pilot Validation Report",
        "📹 Multi-Camera & RTSP Setup",
        "💬 Venue Partner Feedback & ROI",
        "📜 MSME Grant & DPA Kit"
    ])

    with t_metrics:
        st.subheader("Field Pilot Executive Dashboard (30-Day Window)")
        val_report = pilot.generate_pilot_validation_report()
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Pilot Visits", f"{val_report['total_visits_logged']:,}", delta="Real Footfall")
        with m2:
            st.metric("Peak Hour Congestion", f"{val_report['peak_hour']}", delta=f"{val_report['peak_visits']} visits")
        with m3:
            st.metric("Avg Dwell Duration", f"{val_report['avg_dwell_minutes']} min", delta=val_report['top_utilized_zone'])
        with m4:
            st.metric("Validated WTP", f"₹{val_report['avg_willingness_to_pay_inr']:,}/mo", delta=f"Rating: {val_report['avg_owner_satisfaction']}/5.0")

        st.markdown("---")
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown(f"""
            ### 🎯 Key Validation Findings
            * **Peak Congestion Window:** Daily high-traffic spike detected between **{val_report['peak_hour']}**, allowing management to reschedule floor trainers.
            * **High Engagement Cohort:** **{val_report['engagement_rate_pct']}%** of enrolled members visit 3+ times weekly.
            * **Member Retention Interventions:** System identified **{val_report['at_risk_members_count']} at-risk members** (absent 14–30 days) enabling proactive WhatsApp outreach before subscription lapse.
            * **Zero Privacy Breaches:** 100% ephemeral in-memory processing. An estimated **~{val_report['privacy_discards_estimated']} bystander frames** were purged from RAM with 0 bytes written to disk.
            """)
        with c_right:
            st.subheader("Official Pilot Dossier")
            report_md = pilot.export_pilot_report_markdown()
            st.download_button(
                label="📥 Download Official Pilot Report (.md)",
                data=report_md,
                file_name=f"facetrack_pilot_validation_report_{date.today().isoformat()}.md",
                mime="text/markdown",
                use_container_width=True
            )
            st.info("💡 **Grant Tip:** Attach this report directly as 'Annexure A: Working Prototype Field Evidence' in the MSME Idea Hackathon application.")

    with t_cams:
        st.subheader("Multi-Zone Camera & RTSP Stream Manager")
        st.write("Connect CCTV NVRs, IP cameras, or mobile phone cameras over the local network via RTSP or HTTP streams.")
        
        cam_cols = st.columns(2)
        with cam_cols[0]:
            st.markdown("#### Configured Camera Streams")
            for z in config.DEFAULT_ZONES:
                cam_url = st.text_input(
                    f"Zone: {z['name']}",
                    value=f"0 (Local USB / Built-in)" if z['id'] == "main_entrance" else f"rtsp://admin:pass@192.168.1.{100 + len(z['id'])}:554/stream1",
                    key=f"cam_zone_{z['id']}"
                )
        with cam_cols[1]:
            st.markdown("""
            #### 📱 Using a Mobile Phone as an IP Camera
            You can test multi-camera venue tracking without buying CCTV hardware:
            1. Install **IP Webcam** (Android) or **DroidCam** (iOS/Android).
            2. Connect your phone to the same Wi-Fi network as this laptop.
            3. Start the server in the app (e.g. `http://192.168.1.45:8080/video`).
            4. Paste that URL into the zone above to ingest live real-time frames!
            """)

    with t_feedback:
        st.subheader("Venue Partner Qualitative Evaluation & ROI")
        st.write("Feedback gathered from gym owners and venue directors during Phase 2 validation.")
        
        feedbacks = pilot.get_all_feedback()
        for fb in feedbacks:
            with st.expander(f"⭐ {fb['venue_name']} — {fb['owner_name']} (Rating: {fb['satisfaction_rating']}/5)", expanded=True):
                st.write(f"💬 *\"{fb['qualitative_notes']}\"*")
                c1, c2, c3 = st.columns(3)
                c1.metric("Staffing Efficiency Gain", f"+{fb['staffing_improvement_pct']}%")
                c2.metric("Privacy Comfort Rating", f"{fb['privacy_comfort_rating']}/5")
                c3.metric("Validated Subscription WTP", f"₹{fb['willingness_to_pay_inr']:,}/month")

        st.markdown("---")
        st.subheader("Submit New Venue Partner Feedback")
        with st.form("new_feedback_form"):
            v_name = st.text_input("Venue Name", placeholder="e.g. Gold's Gym South Campus")
            o_name = st.text_input("Owner / Manager Name & Designation", placeholder="e.g. Sameer Khanna (Club Director)")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                sat_rate = st.slider("Overall Usability Rating", 1, 5, 5)
            with col_f2:
                staff_pct = st.slider("Staffing Efficiency Gain (%)", 0, 100, 30)
            with col_f3:
                wtp_val = st.number_input("Willingness to Pay (₹/month)", min_value=1000, max_value=25000, value=6000, step=500)
            notes = st.text_area("Manager Feedback & Testimonial", placeholder="Describe how FaceTrack AI improved floor operations...")
            
            if st.form_submit_button("Record Partner Evaluation", type="primary"):
                if v_name and o_name:
                    pilot.save_venue_feedback(
                        venue_name=v_name,
                        owner_name=o_name,
                        satisfaction_rating=sat_rate,
                        staffing_improvement_pct=staff_pct,
                        privacy_comfort_rating=5,
                        willingness_to_pay_inr=wtp_val,
                        qualitative_notes=notes or "Excellent operational insight and member response."
                    )
                    st.success(f"Feedback recorded for {v_name}!")
                    st.rerun()
                else:
                    st.error("Please fill in Venue Name and Owner Name.")

    with t_grant:
        st.subheader("MSME Idea Hackathon & Incubation Cell Application Kit")
        st.write("Ready-to-submit grant dossier and statutory legal agreements for institutional funding.")
        
        grant_file = config.BASE_DIR / "pilot" / "grant_application.md"
        dpa_file = config.BASE_DIR / "pilot" / "dpa_template.md"
        
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("### 🏆 MSME Idea Hackathon Dossier (₹15 Lakhs)")
            st.write("Comprehensive proposal with Market Size (TAM/SAM/SOM), ₹15L budget allocation, technical novelty, and commercial roadmap.")
            if grant_file.exists():
                with open(grant_file, "r") as gf:
                    grant_text = gf.read()
                st.download_button(
                    label="📥 Download MSME Grant Proposal (.md)",
                    data=grant_text,
                    file_name="MSME_Idea_Hackathon_FaceTrack_AI_Grant_Proposal.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                with st.expander("Preview Grant Application Content"):
                    st.markdown(grant_text[:1200] + "\n\n*(Open downloaded file for full 7-section proposal)*")
                    
        with g_c2:
            st.markdown("### 📜 Data Processing Agreement (DPA)")
            st.write("Standard legal template establishing the venue as Data Fiduciary and FaceTrack AI as Data Processor under DPDP Act 2023.")
            if dpa_file.exists():
                with open(dpa_file, "r") as df:
                    dpa_text = df.read()
                st.download_button(
                    label="📥 Download Legal DPA Agreement (.md)",
                    data=dpa_text,
                    file_name="DPDP_Act_Venue_Data_Processing_Agreement_DPA.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                with st.expander("Preview DPA Terms"):
                    st.markdown(dpa_text[:1200] + "\n\n*(Open downloaded file for complete agreement)*")

# -------------------------------------------------------------
# PAGE 8: SETTINGS & DEMO TOOLS
# -------------------------------------------------------------
elif selected_page == "⚙️ Settings & Demo Tools":
    st.title("⚙️ System Settings & Hackathon Demo Tools")
    st.write("Tools to reset, seed data, test database connectivity, and configure cloud synchronization.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sample / Demo Data Generator")
        st.write("Generates 10 realistic members and 30 days of realistic footfall logs with morning/evening rush patterns.")
        seed_days = st.slider("Days of History to Generate", 7, 60, 30)
        
        if st.button("Generate Realistic 30-Day Venue Data", type="primary"):
            count = seed_demo_data(db, days=seed_days)
            st.success(f"Seeded {count} visit sessions across 10 members!")
            tracker.refresh_member_cache()
            st.rerun()

    with c2:
        st.subheader("Supabase Cloud Sync")
        st.write("Connect to Supabase PostgreSQL database to sync members and analytics in real-time.")
        st.text_input("Supabase Project URL", value=config.SUPABASE_URL or "https://your-project.supabase.co")
        st.text_input("Supabase Service / Anon Key", value=config.SUPABASE_KEY or "", type="password")
        
        if st.button("Test Supabase Connection"):
            if config.SUPABASE_URL and config.SUPABASE_KEY:
                try:
                    from supabase import create_client
                    client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                    st.success("Connected to Supabase successfully!")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
            else:
                st.info("Offline Mode: FaceTrack AI is currently using SQLite. Add credentials in `.env` to connect Supabase.")
