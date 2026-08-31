import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Stable parts of the API
base_url = "https://generativelanguage.googleapis.com/v1beta"

# Model is the only variable part
url = f"{base_url}/models/{model}:generateContent"

headers = {
    "x-goog-api-key": api_key,
    "Content-Type": "application/json",
}

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hello are you ready to code?"
                }
            ]
        }
    ],
    "generationConfig": {
        "maxOutputTokens": 4096
    }
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120
)

print(f"Status: {response.status_code}")

if response.ok:
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print("Error:", response.text)