"""
Boot test for Streamlit app
"""
import subprocess
import time
import urllib.request

def test_streamlit_boot():
    p = subprocess.Popen(
        ["python", "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8502"],
        cwd=r"C:\Users\surya\.gemini\antigravity\scratch\ddos-traffic-classification"
    )
    time.sleep(5)
    try:
        with urllib.request.urlopen("http://localhost:8502") as response:
            status = response.getcode()
            print(f"[STREAMLIT BOOT TEST PASSED] HTTP Status: {status}")
    finally:
        p.terminate()
        p.wait()

if __name__ == "__main__":
    test_streamlit_boot()
