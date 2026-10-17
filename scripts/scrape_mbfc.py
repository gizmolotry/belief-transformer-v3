import requests
from bs4 import BeautifulSoup
import re

outlets = [
    ("cnn", "https://mediabiasfactcheck.com/cnn/"),
    ("foxnews", "https://mediabiasfactcheck.com/fox-news/"),
    ("bbc", "https://mediabiasfactcheck.com/bbc/"),
    ("nypost", "https://mediabiasfactcheck.com/new-york-post/"),
    ("dailymail", "https://mediabiasfactcheck.com/daily-mail/"),
    ("guardian", "https://mediabiasfactcheck.com/the-guardian/"),
    ("middleeasteye", "https://mediabiasfactcheck.com/middle-east-eye/"),
    ("israelnatnews", "https://mediabiasfactcheck.com/israel-national-news/"),
    ("ynet", "https://mediabiasfactcheck.com/ynetnews/"),
    ("hindustan", "https://mediabiasfactcheck.com/hindustan-times/"),
    ("aljazeera", "https://mediabiasfactcheck.com/al-jazeera/")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

results = {}
for name, url in outlets:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # MBFC bias score is often formatted as "Bias Rating: LEFT" or similar
        # But they also have a specific float score like "Bias Rating: -3.5" or it's in the text
        text = soup.get_text()
        
        # Look for "Bias Rating:" or "Bias Score:" followed by a number
        # Actually MBFC uses a 0 to 10 scale (or -10 to +10)
        matches = re.findall(r'Bias Rating:.*?([-\d.]+)', text, re.IGNORECASE)
        if not matches:
            matches = re.findall(r'Bias Score:.*?([-\d.]+)', text, re.IGNORECASE)
        
        # Let's just find the text "Bias Rating:" and print the surrounding 50 chars
        idx = text.lower().find("bias rating:")
        if idx != -1:
            snippet = text[idx:idx+100].replace('\n', ' ')
            print(f"{name}: {snippet}")
        else:
            print(f"{name}: Bias Rating not found in text.")
            
    except Exception as e:
        print(f"Failed {name}: {e}")
