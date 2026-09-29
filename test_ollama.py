import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen2.5:3b",
    "prompt": "Explain what an LLM is in one simple sentence.",
    "stream": False
}

response = requests.post(url, json=data)

print(response.json()["response"])