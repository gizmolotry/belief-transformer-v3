"""
Generate an adversarial, out-of-distribution mixed corpus.
Loads the structured Gaza articles (50%) and mixes them with 50% completely
unrelated articles (world economics, astronomy, sports, gardening) to test
the system's false-positive resistance and manifold compression behaviors.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

OOD_TEMPLATES = [
    # Astronomy
    "Astronomers utilizing the James Webb Space Telescope have identified a new exoplanet in the habitable zone of a nearby red dwarf star. The discovery, detailed in a recent journal article, provides critical spectral data about atmospheric composition and the existence of water vapor. Researchers emphasize that future observations will clarify the planet's atmospheric stability and temperature variance.",
    # Gardening
    "Preparing soil for spring planting requires careful attention to organic matter and drainage. Agricultural experts suggest adding compost to heavy clay soils to improve root penetration and nutrient availability. Maintaining a consistent watering schedule during early root development is essential for achieving high yields in vegetable gardens.",
    # Sports
    "The local league championship concluded today with a dramatic final match that went into double overtime. The winning team demonstrated exceptional defensive structure and transition play, securing their victory with a late penalty kick. Analysts praise the team's tactical discipline and physical conditioning throughout the season.",
    # Economics
    "Global market indicators showed modest growth today as central bank representatives discussed inflation targets and monetary policy adjustments. Economists suggest that the stabilization of supply chains has contributed to reduced volatility in commodity indexes. Investors remain cautious as long-term yield curves fluctuate in response to fiscal projections."
]

def generate_ood_article(i: int) -> dict:
    # Build text from OOD templates
    paragraphs = []
    for _ in range(random.randint(4, 6)):
        paragraphs.append(random.choice(OOD_TEMPLATES))
    content = "\n\n".join(paragraphs)
    
    start_date = datetime(2022, 1, 1)
    random_days = random.randint(0, 365 * 2)
    article_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    title_templates = [
        f"Global Science Update: New Exoplanet Discovery",
        f"Spring Gardening Guide: Optimizing Soil Drainage",
        f"Championship Final: Tactical Analysis of the Decisive Match",
        f"Market Report: Central Bank Interest Rate Sweep"
    ]
    
    return {
        "title": random.choice(title_templates),
        "publication": "OOD Monitor",
        "author": "System OOD Generator",
        "timestamp": article_date,
        "event_id": "OOD",
        "perspective_tag": "OOD_Noise",  # Label as OOD_Noise
        "perspective_type": "OOD / Non-Ideological",
        "content": content
    }

def main():
    sythgen_dir = Path(__file__).parent
    structured_file = sythgen_dir / "high_quality_articles.jsonl"
    
    if not structured_file.exists():
        # Fallback: run generate_test_corpus.py to generate it first
        print("high_quality_articles.jsonl not found. Running generate_test_corpus.py...")
        import subprocess
        subprocess.check_call(["python", str(sythgen_dir / "generate_test_corpus.py")])
        
    # Load structured articles
    with open(structured_file, "r", encoding="utf-8") as f:
        structured_articles = [json.loads(line) for line in f]
        
    print(f"Loaded {len(structured_articles)} structured ideological articles.")
    
    # Generate matching number of OOD articles
    n_ood = len(structured_articles)
    print(f"Generating {n_ood} out-of-distribution articles...")
    
    ood_articles = []
    for i in range(n_ood):
        ood_articles.append(generate_ood_article(i))
        
    # Mix them together
    mixed_articles = structured_articles + ood_articles
    
    # Shuffle
    random.seed(42)
    random.shuffle(mixed_articles)
    
    # Write to target path
    output_file = sythgen_dir.parent / "data" / "control_adversarial.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for article in mixed_articles:
            json.dump(article, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"Successfully generated adversarial mixed corpus: {output_file} ({len(mixed_articles)} articles total)")

if __name__ == "__main__":
    main()
