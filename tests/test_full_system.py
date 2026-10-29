"""
End-to-End Comprehensive System Audit & Verification
Simulates real human user interactions across all API endpoints, models, files, and features.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import mimetypes

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def log_test(step_name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {step_name:<45} {details}")
    return passed

def run_tests():
    print("\n" + "="*80)
    print("      🧪 EXHAUSTIVE REAL-WORLD SYSTEM TEST & HUMAN USER AUDIT 🧪      ")
    print("="*80 + "\n")
    
    all_passed = True

    # TEST 1: WEB DASHBOARD ACCESS
    try:
        req = urllib.request.Request(f"{BASE_URL}/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as res:
            html = res.read().decode("utf-8")
            has_title = "DDoS Sentinel SOC" in html
            has_inspector = "Traffic Inspector" in html
            has_stream = "Live Stream Simulator" in html
            p = res.getcode() == 200 and has_title and has_inspector and has_stream
            all_passed &= log_test("1. Web UI Dashboard Access (HTTP 200)", p, f"({len(html):,} bytes received)")
    except Exception as e:
        all_passed &= log_test("1. Web UI Dashboard Access", False, str(e))

    # TEST 2: ONE-CLICK BUTTON - HIGH ATTACK
    try:
        data = urllib.parse.urlencode({"sample": "attack", "model": "Random Forest"}).encode()
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=data)
        with urllib.request.urlopen(req, timeout=15) as res:
            d = json.loads(res.read().decode())
            p = (d.get("total") == 200 and d.get("threat_level") == "CRITICAL" and d.get("ddos_count") >= 195)
            details = f"Flows: {d.get('total')} | DDoS: {d.get('ddos_count')} | Alert: {d.get('threat_level')}"
            all_passed &= log_test("2. 'Load High Attack Traffic' Button", p, details)
    except Exception as e:
        all_passed &= log_test("2. 'Load High Attack Traffic' Button", False, str(e))

    # TEST 3: ONE-CLICK BUTTON - CLEAN NORMAL TRAFFIC
    try:
        data = urllib.parse.urlencode({"sample": "normal", "model": "Random Forest"}).encode()
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=data)
        with urllib.request.urlopen(req, timeout=5) as res:
            d = json.loads(res.read().decode())
            p = (d.get("total") == 200 and d.get("threat_level") == "NORMAL" and d.get("ddos_count") == 0)
            details = f"Flows: {d.get('total')} | Normal: {d.get('normal_count')} | Alert: {d.get('threat_level')}"
            all_passed &= log_test("3. 'Load Clean Normal Traffic' Button", p, details)
    except Exception as e:
        all_passed &= log_test("3. 'Load Clean Normal Traffic' Button", False, str(e))

    # TEST 4: ONE-CLICK BUTTON - MIXED TRAFFIC
    try:
        data = urllib.parse.urlencode({"sample": "mixed", "model": "Random Forest"}).encode()
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=data)
        with urllib.request.urlopen(req, timeout=5) as res:
            d = json.loads(res.read().decode())
            p = (d.get("total") == 300 and d.get("normal_count") > 0 and d.get("ddos_count") > 0)
            details = f"Flows: {d.get('total')} | Normal: {d.get('normal_count')} | DDoS: {d.get('ddos_count')} ({d.get('attack_rate')}%)"
            all_passed &= log_test("4. 'Load Mixed Traffic' Button", p, details)
    except Exception as e:
        all_passed &= log_test("4. 'Load Mixed Traffic' Button", False, str(e))

    # TEST 5: MULTI-MODEL INFERENCE (ALL 5 ALGORITHMS)
    models = ["Random Forest", "Decision Tree", "Logistic Regression", "SVM", "KNN"]
    for m in models:
        try:
            data = urllib.parse.urlencode({"sample": "mixed", "model": m}).encode()
            req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=data)
            with urllib.request.urlopen(req, timeout=5) as res:
                d = json.loads(res.read().decode())
                p = d.get("total") == 300 and "table_rows" in d
                all_passed &= log_test(f"5. Model Engine: {m}", p, f"Verdict rate: {d.get('attack_rate')}%")
        except Exception as e:
            all_passed &= log_test(f"5. Model Engine: {m}", False, str(e))

    # TEST 6: REAL CSV FILE UPLOAD VIA MULTIPART/FORM-DATA
    try:
        sample_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_traffic_attack.csv")
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        with open(sample_file_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="uploaded_test.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + file_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"Random Forest\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=body)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=5) as res:
            d = json.loads(res.read().decode())
            p = d.get("total") == 200 and d.get("threat_level") == "CRITICAL"
            all_passed &= log_test("6. Drag & Drop CSV Upload Inspection", p, f"Processed {d.get('total')} uploaded rows")
    except Exception as e:
        all_passed &= log_test("6. Drag & Drop CSV Upload Inspection", False, str(e))

    # TEST 7: LIVE PACKET STREAM SIMULATOR API
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/stream/sample")
        with urllib.request.urlopen(req, timeout=5) as res:
            d = json.loads(res.read().decode())
            packets = d.get("packets", [])
            p = len(packets) > 0 and "verdict" in packets[0] and "packets_s" in packets[0]
            all_passed &= log_test("7. Live Stream Simulator Telemetry Feed", p, f"Stream buffer: {len(packets)} packets")
    except Exception as e:
        all_passed &= log_test("7. Live Stream Simulator Telemetry Feed", False, str(e))

    # TEST 8: BENCHMARKS COMPARISON API
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/benchmarks")
        with urllib.request.urlopen(req, timeout=5) as res:
            d = json.loads(res.read().decode())
            metrics = d.get("metrics", [])
            p = len(metrics) == 5 and all("Model" in m and "Accuracy" in m for m in metrics)
            all_passed &= log_test("8. Benchmarks API (5 Models Metrics)", p, f"{len(metrics)} algorithms benchmarked")
    except Exception as e:
        all_passed &= log_test("8. Benchmarks API (5 Models Metrics)", False, str(e))

    # TEST 9: DIAGNOSTIC PLOTS & IMAGE SERVING
    plots = ["confusion_matrices.png", "roc_curves.png", "model_comparison.png", "top_features.png"]
    for img_name in plots:
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/charts/{img_name}")
            with urllib.request.urlopen(req, timeout=5) as res:
                img_data = res.read()
                p = res.getcode() == 200 and len(img_data) > 10000
                all_passed &= log_test(f"9. Image Asset: {img_name}", p, f"({len(img_data):,} bytes)")
        except Exception as e:
            all_passed &= log_test(f"9. Image Asset: {img_name}", False, str(e))

    # TEST 10: AUDIT CSV EXPORT
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/export")
        with urllib.request.urlopen(req, timeout=5) as res:
            csv_data = res.read().decode("utf-8")
            p = res.getcode() == 200 and "Prediction" in csv_data and "Confidence" in csv_data
            all_passed &= log_test("10. Audit Report CSV Export", p, f"({len(csv_data):,} bytes)")
    except Exception as e:
        all_passed &= log_test("10. Audit Report CSV Export", False, str(e))

    print("\n" + "="*80)
    if all_passed:
        print("  🎉 VERDICT: 100% FULLY FUNCTIONAL, FULLY TESTED, AND READY FOR USE! 🎉  ")
    else:
        print("  ⚠️ SOME TESTS FAILED. CHECK DETAILS ABOVE.  ")
    print("="*80 + "\n")
    return all_passed

if __name__ == "__main__":
    run_tests()
