import requests
import json

# Your live URL
url = "https://resnet-classifier-aditya.onrender.com/predict"

# Create a list of 3,072 dummy pixel values (3 * 32 * 32)
# This simulates a blank gray image
dummy_data = [0.5] * 3072

payload = {"data": dummy_data}
headers = {"Content-Type": "application/json"}

print("Sending request to microservice...")
response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    print("✅ Success!")
    print("Prediction:", response.json())
else:
    print(f"❌ Failed with status code: {response.status_code}")
    print("Error details:", response.text)