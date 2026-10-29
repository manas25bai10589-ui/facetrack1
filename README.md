# FaceTrack AI 🛡️👁️

> **A Privacy-Compliant, Opt-In Facial Recognition Analytics Platform for B2B Venues**  
> *Target Tracks: MSME Idea Hackathon · Campus Incubation Cell · Academic Capstone Project*

---

## 📌 Executive Summary

Modern high-ticket membership venues (gyms, private athletic clubs, luxury retail showrooms) lack granular operational visibility into **peak hours**, **zone-wise congestion**, **equipment dwell times**, and **churn risk**. 

Traditional public camera surveillance violates **India's Digital Personal Data Protection Act, 2023 (DPDP Act)** because it captures biometric data of non-consenting individuals. 

**FaceTrack AI solves this through a B2B Opt-In Pivot:**
- Deployed exclusively in **closed, membership-based venues** where the venue acts as the **Data Fiduciary**.
- Members explicitly sign an **opt-in consent notice** during onboarding as part of their membership benefits (e.g., touchless check-in, VIP hospitality, personalized perks).
- **Privacy-by-Design Engine:** Camera frames are analyzed in volatile RAM. Unmatched/non-consenting faces are **instantly discarded** in-memory. Zero raw video is ever saved to disk.
- Fully supports the statutory **Right to Erasure (Section 12, DPDP Act)** with 1-click permanent purging of biometric embeddings and audit logging.

---

## 🏛️ Technical Architecture

```
                      [ LIVE CAMERA FEED ]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       OpenCV YuNet Face Detector (ONNX)       │
        │     - Real-time bounding box detection        │
        │     - 5 facial landmark alignment            │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        OpenCV SFace Embedder (ONNX)          │
        │     - Extracts 128-dimensional vector        │
        │     - L2-normalized cosine feature space     │
        └──────────────────────┬───────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [ Matched Member? ]            [ Unmatched Face? ]
               │                               │
       YES (Score >= 0.40)            NO (Non-Member)
               │                               │
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │  Log Zone Visit &    │        │  IN-MEMORY DISCARD   │
    │  Dwell Analytics     │        │  - 0 Bytes Persisted │
    │  (SQLite / Supabase) │        │  - Frame Dropped     │
    └──────────────────────┘        └──────────────────────┘
```

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Video Acquisition & Detection** | Python, OpenCV, YuNet ONNX | Ultra-fast CPU/GPU face detection with 5-point facial landmarks |
| **Biometric Embedding** | OpenCV SFace ONNX | 128-dimensional normalized embedding vectors for accurate cosine matching |
| **Backend & Storage** | SQLite (Offline-First) + Supabase | Local database with zero setup, with instant Supabase cloud synchronization |
| **Consent & Compliance** | Custom DPDP Engine | Notice versioning, explicit digital signature logging, Right to Erasure execution |
| **Venue Analytics Dashboard** | Streamlit, Plotly, Pandas | Real-time live HUD, footfall heatmaps, dwell times, and churn risk radar |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.12 verified)
- Webcam (built-in or USB)

### 2. Launching the Platform

You can launch the dashboard with a single command:

```powershell
python -m streamlit run app.py
```

Or double-click `run_app.bat`.

Once launched, open your browser at `http://localhost:8501`.

---

## 🌟 Key Modules & Features

### 1. 🎥 Live Venue Monitor
- Real-time video processing with interactive camera selector.
- Live bounding box HUD:
  - **Emerald Green:** Consented Member recognized with similarity score, tier badge, and real-time dwell timer.
  - **Neutral Gray:** Non-enrolled individual with explicit overlay: `Unenrolled - Frame Discarded`.
- Real-time floor occupancy counter.

### 2. 📊 Footfall & Peak Hours Analytics
- Hourly footfall distribution bar charts and heatmaps across last 30 days.
- Peak hour detection (e.g. 07:00–09:00 AM & 18:00–20:00 PM).
- Automated staffing recommendations for gym trainers, reception staff, and maintenance.

### 3. ⏱️ Dwell Time & Zone Utilization
- Zone-wise breakdown (Cardio Zone, Free Weights, VIP Lounge, Studio Area, Main Entrance).
- Average dwell time per zone in minutes.
- Congestion radar to prevent equipment bottlenecks.

### 4. 🔄 Member Retention & Churn Risk Engine
- Automated categorisation:
  - **Frequent Attendees** (>3 visits/week)
  - **Moderate Attendees** (1-2 visits/week)
  - **⚠️ At-Risk Members** (Absent for 14–30 days) — generates proactive re-engagement alerts.
  - **❌ Churned Members** (>30 days absent).

### 5. 👤 Member Enrollment & Digital Consent Portal
- Onboard new members in seconds:
  - Capture photo via webcam or file upload.
  - Generates 128-d face embedding automatically.
  - Collects digitally signed opt-in consent waiver with timestamped audit trail.

### 6. 🛡️ DPDP Act 2023 Compliance & Erasure Center
- **Right to Erasure (Section 12):** 1-click permanent purge of a member's biometric embedding.
- **Audit Trails:** Immutable log of all consent grants, withdrawals, and purges.
- **Audit Export:** 1-click download of compliance report in CSV format for regulatory review.

### 7. ⚙️ Hackathon Demo & Presentation Tools
- 1-click realistic synthetic data generator (seeds 30 days of realistic footfall curves and 10 member profiles).
- Instant offline/online operation.

---

## ⚖️ Legal & DPDP Act 2023 Alignment

| DPDP Act Requirement | FaceTrack AI Implementation |
|---|---|
| **Notice & Consent (Section 5 & 6)** | Standardized opt-in consent waiver signed during onboarding before any embedding is saved. |
| **Purpose Limitation** | Data is strictly utilized for venue dwell analytics and access; never sold or shared. |
| **Storage Limitation** | Raw camera frames are processed in-memory and immediately destroyed; only vectors persist. |
| **Right to Correction & Erasure (Section 12)** | Dedicated compliance center allows immediate deletion of biometrics upon member request. |
| **Data Fiduciary Accountability** | Automated immutable audit logs track all consent actions with ISO timestamps. |

---


