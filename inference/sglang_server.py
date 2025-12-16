import requests

URL = "http://127.0.0.1:30000/v1/completions/"

payload = {
    "model": "moegpt",
    "prompt": "def binary_search(arr, target):\n    ",
    "max_tokens": 120,
    "temperature": 0.6,
}

s = requests.Session()
s.trust_env = False   # <-- this is the "no-proxy" switch

r = s.post(URL, json=payload, timeout=120)
r.raise_for_status()
print(r.json()["choices"][0]["text"])
