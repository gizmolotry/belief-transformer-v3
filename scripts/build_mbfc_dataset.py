import requests
from bs4 import BeautifulSoup
import re
import json

URLS = {
    "israelnationalnews.com": "https://mediabiasfactcheck.com/israel-national-news/",
    "ynetnews.com": "https://mediabiasfactcheck.com/ynet-news/",
    "presstv.ir": "https://mediabiasfactcheck.com/press-tv/",
    "aa.com.tr": "https://mediabiasfactcheck.com/anadolu-agency/",
    "theguardian.com": "https://mediabiasfactcheck.com/the-guardian/",
    "hindustantimes.com": "https://mediabiasfactcheck.com/hindustan-times/",
    "middleeasteye.net": "https://mediabiasfactcheck.com/middle-east-eye/",
    "middleeastmonitor.com": "https://mediabiasfactcheck.com/middle-east-monitor/",
    "algemeiner.com": "https://mediabiasfactcheck.com/the-algemeiner/",
    "english.news.cn": "https://mediabiasfactcheck.com/xinhua-news-agency/",
    "thefrontierpost.com": "https://mediabiasfactcheck.com/the-frontier-post/",
    "marketscreener.com": "https://mediabiasfactcheck.com/market-screener/",
    "english.aawsat.com": "https://mediabiasfactcheck.com/asharq-al-awsat/",
    "english.wafa.ps": "https://mediabiasfactcheck.com/wafa/",
    "aljazeera.com": "https://mediabiasfactcheck.com/al-jazeera/",
    "cnn.com": "https://mediabiasfactcheck.com/cnn/",
    "foxnews.com": "https://mediabiasfactcheck.com/fox-news/",
    "bbc.com": "https://mediabiasfactcheck.com/bbc/",
    "dailymail.co.uk": "https://mediabiasfactcheck.com/daily-mail/",
    "nypost.com": "https://mediabiasfactcheck.com/new-york-post/",
    "reuters.com": "https://mediabiasfactcheck.com/reuters/",
    "apnews.com": "https://mediabiasfactcheck.com/associated-press/",
    "politico.com": "https://mediabiasfactcheck.com/politico/",
    "breitbart.com": "https://mediabiasfactcheck.com/breitbart/",
    "washingtonpost.com": "https://mediabiasfactcheck.com/washington-post/",
    "nytimes.com": "https://mediabiasfactcheck.com/new-york-times/",
    "wsj.com": "https://mediabiasfactcheck.com/wall-street-journal/",
    "npr.org": "https://mediabiasfactcheck.com/npr/",
    "bloomberg.com": "https://mediabiasfactcheck.com/bloomberg/",
    "cnbc.com": "https://mediabiasfactcheck.com/cnbc/"
}

def main():
    headers = {"User-Agent": "Mozilla/5.0"}
    matched = {}
    for pub, url in URLS.items():
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                text = BeautifulSoup(r.text, 'html.parser').get_text()
                matches = re.findall(r'Bias Rating:.*?([-\d.]+)', text, re.IGNORECASE)
                if not matches:
                    matches = re.findall(r'Bias Score:.*?([-\d.]+)', text, re.IGNORECASE)
                if matches:
                    score = float(matches[0])
                    print(f"{pub}: {score}")
                    matched[pub] = score
        except Exception as e:
            pass
            
    with open('data/mbfc_continuous_scores.json', 'w') as f:
        json.dump(matched, f, indent=2)
    print(f"\nScraped {len(matched)} scores.")

if __name__ == '__main__':
    main()
