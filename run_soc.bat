@echo off
title DDoS Sentinel Security Operations Center (SOC)
color 0B
cd /d "C:\Users\surya\.gemini\antigravity\scratch\ddos-traffic-classification"
echo ==============================================================================
echo        DDOS SENTINEL - SECURITY OPERATIONS CENTER (FLASK SOC WEB APP)
echo ==============================================================================
echo.
echo [1/2] Starting High-Performance Flask Web Server on Port 5000...
echo [2/2] Opening Browser to http://localhost:5000 ...
start "" "http://localhost:5000"
python web_app.py
pause
