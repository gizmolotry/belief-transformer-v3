import csv
import json
import random

# ==============================================================================
# PUBLISHER SCORES (Test A)
# ==============================================================================
# Sourced from AllSides Media Bias Ratings (July 2026 snapshot) and Media Bias/Fact Check (MBFC)
# Scale: -2 (Far Left/Left), -1 (Lean Left / Left-Center), 0 (Center), 1 (Lean Right / Right-Center), 2 (Right / Far Right)
PUBLISHER_BIAS_DICT = {
    'theguardian.com': -2,        # Source: AllSides (Left)
    'middleeasteye.net': -2,      # Source: AllSides (Left)
    'nypost.com': 2,              # Source: AllSides (Right - News & Opinion)
    'dailymail.co.uk': 1,         # Source: AllSides (Lean Right)
    'israelnationalnews.com': 1,  # Source: MBFC (Right-Center)
    'ynetnews.com': -1,           # Source: MBFC (Left-Center)
    'hindustantimes.com': 1,      # Source: MBFC (Right-Center)
    'aljazeera.com': -1,          # Source: AllSides (Lean Left / Left-Center)
    'foxnews.com': 2,             # Source: AllSides (Right)
    'bbc.com': 0,                 # Source: AllSides (Center)
    'bbc.co.uk': 0,               # Source: AllSides (Center)
    'cnn.com': -2                 # Source: AllSides (Left)
}

def build_publisher_scores():
    print("Building publisher_scores.csv...")
    with open('publisher_scores.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['publisher', 'allsides_score'])
        for pub, score in PUBLISHER_BIAS_DICT.items():
            writer.writerow([pub, score])
    print("Done.\n")

# ==============================================================================
# HUMAN CODING SUBSAMPLE (Test B)
# ==============================================================================
def build_human_coding_subsample():
    print("Extracting 30 articles for human coding subsample...")
    articles = []
    with open('data/real_corpus.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('publisher') in PUBLISHER_BIAS_DICT:
                    articles.append(data)
            except:
                pass
                
    if len(articles) == 0:
        print("No articles matched known publishers.")
        return

    random.seed(42) # Deterministic sample
    sample = random.sample(articles, min(30, len(articles)))
    
    with open('human_tension_rating_subsample.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['article_id', 'publisher', 'title', 'content', 'human_tension_score_0_to_4'])
        for idx, article in enumerate(sample):
            # Using internal id or a generated one
            art_id = article.get('id', f'art_{idx}')
            writer.writerow([art_id, article.get('publisher', ''), article.get('title', ''), article.get('content', ''), ''])
            
    print(f"Done. Wrote {len(sample)} articles to human_tension_rating_subsample.csv\n")
    print("NOTE: You must manually fill out the 'human_tension_score_0_to_4' column using the blind prompt.")

if __name__ == '__main__':
    build_publisher_scores()
    build_human_coding_subsample()
