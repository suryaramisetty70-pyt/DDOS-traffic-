@echo off
title DDoS Traffic Classification & Real-Time Alert System
color 0A
cd /d "C:\Users\surya\.gemini\antigravity\scratch\ddos-traffic-classification"
echo ==============================================================================
echo       DDOS TRAFFIC CLASSIFICATION & REAL-TIME ALERT SYSTEM (SOC)
echo ==============================================================================
echo.
echo [1/2] Running end-to-end verification and model inference check...
python src\demo_run.py
echo.
echo [2/2] Launching Streamlit Web Dashboard...
echo Opening browser at http://localhost:8501 ...
echo.
python -m streamlit run app.py
pause
