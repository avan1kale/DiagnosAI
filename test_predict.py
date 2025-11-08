import requests
import json

# Flask backend URL
url = "http://127.0.0.1:5000/predict"

# Sample data (personal info + cancer cell features)
data = {
    "personal": {
        "name": "Avani Kale",
        "age": 45,
        "gender": "Female"
    },
    "features": [
        17.99, 10.38, 122.8, 1001.0,
        0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053,
        8.589, 153.4, 0.006399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654,
        0.4601, 0.1189
    ]
}
data2= {
    "personal": {
        "name": "John Doe",
        "age": 50,
        "gender": "Male"    
    },
    "features": [
        15.99, 10.38, 122.8, 1001.0,
        0.1184, 0.2776, 0.1001, 0.1991,
        0.2419, 0.07871, 1.095, 0.9053,
        9.589, 163.4, 0.004399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193,
        27.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654,
        0.4601, 0.1159
    ]
}
# Send POST request to Flask backend
response = requests.post(url, json=data)

# Print result
if response.status_code == 200:
    print("✅ Response from Flask API:")
    print(json.dumps(response.json(), indent=4))
else:
    print("❌ Error:", response.status_code)
    print(response.text)

# Send another POST request to Flask backend
response = requests.post(url, json=data2)
if response.status_code == 200:
    print("✅ Response from Flask API:")
    print(json.dumps(response.json(), indent=4))
else:
    print("❌ Error:", response.status_code)
    print(response.text)