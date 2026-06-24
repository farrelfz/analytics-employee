import urllib.request

try:
    # Set timeout to 5 seconds
    response = urllib.request.urlopen("http://localhost:8501", timeout=5)
    print("Health Check SUCCESS!")
    print("Streamlit Response Status:", response.status)
    print("Headers:\n", response.headers)
except Exception as e:
    print("Health Check FAILED:", e)