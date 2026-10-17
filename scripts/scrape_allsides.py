import requests
from bs4 import BeautifulSoup
import re

sources = [
    ("cnn", "https://www.allsides.com/news-source/cnn-media-bias"),
    ("fox-news", "https://www.allsides.com/news-source/fox-news-media-bias"),
    ("bbc", "https://www.allsides.com/news-source/bbc-news-media-bias"),
    ("new-york-post", "https://www.allsides.com/news-source/new-york-post"),
    ("daily-mail", "https://www.allsides.com/news-source/daily-mail"),
    ("the-guardian", "https://www.allsides.com/news-source/the-guardian")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

for name, url in sources:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Look for the numerical rating which is often in a specific div or script
        # Sometimes it's in a text string like "rating of -2.37"
        text = soup.get_text()
        
        # Regex to find something like "rating of -1.5" or similar
        matches = re.findall(r'numerical rating of ([-0-9.]+)', text, re.IGNORECASE)
        if not matches:
            matches = re.findall(r'Bias Meter.*?([-0-9.]+)', text, re.IGNORECASE)
            
        print(f"--- {name} ---")
        if matches:
            print(f"Found explicit scores: {matches}")
        else:
            print("No explicit float found in text.")
            
    except Exception as e:
        print(f"Failed {name}: {e}")
