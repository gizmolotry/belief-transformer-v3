import requests
import json
import re

url = "https://adfontesmedia.com/interactive-media-bias-chart/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

r = requests.get(url, headers=headers, timeout=10)
print(f"Status: {r.status_code}")

# Let's search for JSON data embedded in the page
matches = re.findall(r'var\s+chartData\s*=\s*(\[.*?\]);', r.text, re.DOTALL)
if matches:
    print("Found chart data!")
    data = json.loads(matches[0])
    print(f"Number of sources: {len(data)}")
else:
    print("No chart data array found in page.")
    
# Or maybe there's a specific API endpoint?
