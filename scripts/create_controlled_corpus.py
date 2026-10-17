import json
import csv
from collections import defaultdict
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    scores_path = root / 'publisher_scores.csv'
    corpus_path = root / 'data' / 'real_corpus.jsonl'
    out_path = root / 'data' / 'controlled_N60_corpus.jsonl'
    
    # Load all rated publishers
    scores = {}
    with open(scores_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row['publisher']] = row['allsides_score']
            
    print(f"Found {len(scores)} rated publishers in dictionary.")
    
    # Extract exactly 5 articles per publisher
    articles_by_pub = defaultdict(list)
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pub = data.get('publication', data.get('source', 'Unknown'))
            # Fix text extraction schema here as well, although data_utils handles it
            text = data.get('content') or data.get('text') or data.get('body') or ""
            
            # Require minimum length to ensure valid semantic text
            if pub in scores and len(articles_by_pub[pub]) < 5 and len(text.strip()) > 500:
                # Store the standard structure for data_utils
                cleaned_article = {
                    'publication': pub,
                    'content': text.strip(),
                    'allsides_score': scores[pub]
                }
                articles_by_pub[pub].append(cleaned_article)
                
            # Stop if we have 5 from all 12
            if len(articles_by_pub) == 12 and all(len(v) == 5 for v in articles_by_pub.values()):
                break

    # Flatten and save
    out_articles = []
    for pub, articles in articles_by_pub.items():
        out_articles.extend(articles)
        print(f"Publisher: {pub} -> {len(articles)} articles (Score: {scores[pub]})")
        
    print(f"Total articles extracted: {len(out_articles)}")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for a in out_articles:
            f.write(json.dumps(a) + '\n')
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
