import requests

# API endpoint
url = "http://127.0.0.1:8000/ask"

# Question to ask
payload = {
    "question": "What is the main purpose of the document?"
}

# Send POST request
response = requests.post(url, json=payload)

# Display result
if response.status_code == 200:
    print("✅ Response from API:")
    print(response.json())
    print(response.json()["response"])
else:
    print("❌ Failed to get response:")
    print(response.text)
