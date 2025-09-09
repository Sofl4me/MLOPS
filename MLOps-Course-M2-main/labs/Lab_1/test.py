import requests
import json

# Define the endpoint URL
url = 'http://localhost:5001/predict'

# Example feature set for a prediction
sample_data = {
    'features': [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
                 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601,
                 0.1189]
}
# Convert the data to JSON format
headers = {'Content-Type': 'application/json'}
data = json.dumps(sample_data)

# Send a POST request
response = requests.post(url, headers=headers, data=data)


# Display the response from the server
print(f"Response Status Code: {response.status_code}")
print(f"Prediction: {response.json()['prediction']}")