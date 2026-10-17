import json

def main():
    with open('data/mbfc_continuous_scores.json', 'r') as f:
        scores = json.load(f)
        
    print(f"Loaded {len(scores)} continuous scores.")
    
    needed_per_pub = 20
    pub_counts = {pub: 0 for pub in scores}
    selected_articles = []
    
    with open('data/real_corpus.jsonl', 'r', encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            pub = data.get('publication') or data.get('source')
            if pub in pub_counts and pub_counts[pub] < needed_per_pub:
                selected_articles.append(data)
                pub_counts[pub] += 1
                
            if all(v == needed_per_pub for v in pub_counts.values()):
                break
                
    # Keep only publishers that actually got 20
    complete_pubs = [p for p, c in pub_counts.items() if c == needed_per_pub]
    print(f"Found {needed_per_pub} articles for {len(complete_pubs)} publishers.")
    
    final_articles = [a for a in selected_articles if (a.get('publication') or a.get('source')) in complete_pubs]
    
    with open('data/controlled_N_dense_corpus.jsonl', 'w', encoding='utf-8') as f:
        for a in final_articles:
            f.write(json.dumps(a) + '\n')
            
    print(f"Wrote {len(final_articles)} articles to data/controlled_N_dense_corpus.jsonl")

if __name__ == '__main__':
    main()
