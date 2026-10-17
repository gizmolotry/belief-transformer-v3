"""
Generate a non-circular unstructured synthetic corpus for baseline testing.
Creates articles representing 5 distinct publications covering varied events,
using templates with general news vocabulary that do NOT align with the 8 V-Observer prompts.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

PUBLICATIONS = [
    {
        "pub": "CBN News",
        "templates": [
            "The recent events in {location} regarding the {event} have sparked widespread discussion in Washington. Prominent faith leaders and policy analysts emphasize that historical context must guide international diplomacy. Observers continue to monitor how these developments will shape regional alliances and faith-based advocacy.",
            "Reports on the {event} highlight the strategic and moral dimension of the situation in {location}. Speaking at a recent conference, representatives urged Western nations to maintain long-term support and recognize the cultural ties that bind our societies to the region.",
            "Diplomatic focus shifted today toward the ongoing {event}. Commentators suggest that geopolitical stability in {location} depends heavily on upholding shared moral commitments and reinforcing historical agreements against rising regional friction."
        ]
    },
    {
        "pub": "Ynet (Security Desk)",
        "templates": [
            "IDF operations surrounding the {event} in {location} are being reviewed by military planners. Operational reports indicate that the tactical environment remains highly volatile, requiring localized adjustments to ensure troop safety and security of vital corridors.",
            "A review of the {event} highlights technical and operational challenges in {location}. Strategic analysts suggest that the timing of these measures is driven by recent shifts in cross-border coordination and the necessity of maintaining systemic stability.",
            "Defense officials reported that security operations during the {event} are progressing as scheduled. Strategic planners emphasize that infrastructure and control checkpoints in {location} remain critical to preventing further operational degradation."
        ]
    },
    {
        "pub": "Haaretz",
        "templates": [
            "The government's policy regarding the {event} in {location} has drawn sharp criticism from legal experts and civil society groups. Analysts argue that bypassing international oversight sets a dangerous precedent and alienates key diplomatic partners.",
            "An editorial on the {event} cautions that short-term policies in {location} carry significant long-term costs. Legal scholars warn of the institutional consequences of prioritizing political expediency over established legal frameworks.",
            "Discussions on the {event} continue in parliamentary committees. Proponents of diplomatic engagement argue that sustainable solutions in {location} require a return to rule-governed negotiations and constitutional accountability."
        ]
    },
    {
        "pub": "Wafa News Agency",
        "templates": [
            "Official sources commented on the ongoing {event} in {location}, calling on global institutions to enforce existing resolutions. The ministry emphasized that unilateral changes to local geographic boundaries undermine future negotiations.",
            "A statement regarding the {event} in {location} warns of the humanitarian impact of recent administrative measures. Local coordinators urged immediate diplomatic pressure to reopen access routes and protect public infrastructure.",
            "The executive committee met today to address the {event}. Representatives stressed that the path to a sustainable resolution in {location} requires full compliance with international conventions and the recognition of historic national rights."
        ]
    },
    {
        "pub": "Al Jazeera",
        "templates": [
            "Coverage of the {event} in {location} highlights the deep historical context of the current regional dynamics. Commentators stress that ignoring the systemic grievances of the population only exacerbates the ongoing crisis.",
            "A panel discussion on the {event} emphasized that regional solidarity remains the primary driver of public opinion. Panelists criticized Western diplomatic responses, arguing they fail to address the core issues of justice and historic displacement in {location}.",
            "Local reporting from {location} indicates that the impact of the {event} is felt across neighboring countries. Regional analysts call for a unified diplomatic front to address the humanitarian fallout and push for systemic reforms."
        ]
    }
]

EVENTS = [
    "administrative zone restructuring",
    "border corridor monitoring",
    "diplomatic treaty review",
    "access checkpoint adjustments",
    "agricultural land zoning",
    "water resource distribution",
    "bilateral trade discussions",
    "municipal election organization"
]

LOCATIONS = [
    "the southern valley region",
    "the eastern border sectors",
    "the central administrative district",
    "the northern transit corridor",
    "the coastal plain zone",
    "the historic crossing point"
]

AUTHORS = [
    "A. Davidson", "Dr. Y. Cohen", "E. Goldberg", "H. Mansour", "K. Al-Sayed",
    "M. Jenkins", "S. Levy", "N. Stern", "R. Hagee", "A. Bohbot"
]

def generate_article(pub_idx: int, article_num: int) -> dict:
    pub_data = PUBLICATIONS[pub_idx]
    event = random.choice(EVENTS)
    location = random.choice(LOCATIONS)
    
    # Fill templates with unstructured news vocabulary
    vocab_map = {
        "event": event,
        "location": location
    }
    
    paragraphs = []
    for _ in range(random.randint(4, 6)):
        template = random.choice(pub_data["templates"])
        para = template.format(**vocab_map)
        paragraphs.append(para)
        
    content = "\n\n".join(paragraphs)
    
    start_date = datetime(2022, 1, 1)
    random_days = random.randint(0, 365 * 2)
    article_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    author = random.choice(AUTHORS)
    
    title_templates = [
        f"{pub_data['pub']} Report: The {event.title()} in {location.title()}",
        f"Perspectives on the {event.title()} Crisis",
        f"Developments regarding the {event.title()} in {location.title()}",
        f"Analysis of the {event.title()} Administrative Action"
    ]
    
    return {
        "title": random.choice(title_templates),
        "publication": pub_data["pub"],
        "author": author,
        "timestamp": article_date,
        "event_id": event[:5].replace(" ", ""),
        "perspective_tag": pub_data["pub"],  # Store pub name as tag to evaluate NMI against pub
        "perspective_type": "Unstructured / Neutral",
        "content": content
    }

def main():
    output_file = Path(__file__).parent.parent / "data" / "control_unstructured.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    articles_per_pub = 16  # Total = 80 articles
    
    print(f"Generating {articles_per_pub * len(PUBLICATIONS)} unstructured non-circular articles...")
    
    articles = []
    for pub_idx in range(len(PUBLICATIONS)):
        for i in range(articles_per_pub):
            article = generate_article(pub_idx, i)
            articles.append(article)
            
    # Shuffle for mixed ordering
    random.seed(42)
    random.shuffle(articles)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"Successfully generated non-circular unstructured corpus: {output_file}")

if __name__ == "__main__":
    main()
