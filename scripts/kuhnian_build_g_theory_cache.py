#!/usr/bin/env python3
"""Build frozen prompt banks for Replicated Generalizability-Theory Design.

Defines 5 conceptual frameworks x 3 independent realizations = 15 banks.
Extracts forward-pass logits using microsoft/deberta-v2-xlarge-mnli on RTX 5070 Ti
for any realizations not already present in cache_adversarial_kuhnian_logits.pt.
Saves:
- config/prompt_banks/replicated_g_theory_banks.json
- data/cache_replicated_g_theory_logits.pt
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import yaml

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# 1. DEFINE FROZEN 15-BANK REPLICATED G-THEORY REGISTRY
# -----------------------------------------------------------------------------

REPLICATED_FRAMEWORKS = {
    "geopolitical_conflict": {
        "name": "Geopolitical Conflict & Territorial Sovereignty",
        "description": "Evaluates discourse through national sovereignty, border defense, state aggression, and territorial occupation.",
        "realizations": {
            "realization_1": {
                "source": "q0_original_geopolitical",
                "label": "Original Geopolitical (Q0)",
                "pairs": None,
            },
            "realization_2": {
                "source": "q1_paraphrase_control",
                "label": "Paraphrase A (Q1)",
                "pairs": None,
            },
            "realization_3": {
                "source": "q_paraphrase_b",
                "label": "Paraphrase B (Independent Syntactic Realization)",
                "pairs": None,
            },
        },
    },
    "humanitarian_civilian_welfare": {
        "name": "Humanitarian Rights & Civilian Protection",
        "description": "Evaluates discourse through human suffering, civilian displacement, starvation, medical neutrality, and basic human rights.",
        "realizations": {
            "realization_1": {
                "source": "q2_humanitarian_rights",
                "label": "Humanitarian Rights (Q2)",
                "pairs": None,
            },
            "realization_2": {
                "source": "extreme_political_ontologies:humanitarian_welfare",
                "label": "Humanitarian Welfare & Suffering",
                "pairs": None,
            },
            "realization_3": {
                "source": "humanitarian_protection_b3",
                "label": "Civilian Harm & Relief Logistics (Realization 3)",
                "pairs": [
                    [
                        "The text details catastrophic civilian death tolls, widespread famine, and the destruction of residential neighborhoods.",
                        "The text emphasizes precise surgical strikes designed to minimize civilian exposure and collateral damage."
                    ],
                    [
                        "The text highlights the total collapse of acute healthcare infrastructure and catastrophic shortages of medical supplies.",
                        "The text highlights operational military field hospitals and ongoing coordination of essential emergency medical transfers."
                    ],
                    [
                        "The text documents severe acute child malnutrition, famine alerts, and the systemic denial of humanitarian food convoys.",
                        "The text documents large-scale international food aid distributions, flour deliveries, and operational bakeries."
                    ],
                    [
                        "The text emphasizes the mass displacement of vulnerable families forced into crowded, unsanitary tent camps.",
                        "The text emphasizes designated civilian safe corridors and temporary evacuation warnings issued for family protection."
                    ],
                    [
                        "The text reports direct artillery and airstrike bombardment impacting designated civilian shelters and schools.",
                        "The text reports that armed insurgent operatives were utilizing designated civilian shelters as operational command outposts."
                    ],
                    [
                        "The text condemns the complete cutoff of municipal drinking water, electrical power grids, and sanitation infrastructure.",
                        "The text notes that secondary utility pipeline repairs and emergency diesel fuel shipments were permitted to proceed."
                    ],
                    [
                        "The text underscores severe psychological trauma, generational grief, and the orphan crisis inflicted on youth.",
                        "The text underscores civil defense rescue efforts, community resilience, and psychological support services."
                    ],
                    [
                        "The text decries unlawful obstructions and lethal strikes against marked humanitarian aid workers and ambulance teams.",
                        "The text states that military convoys implemented strict deconfliction protocols to ensure humanitarian movement."
                    ]
                ],
            },
        },
    },
    "legal_ihl_accountability": {
        "name": "Legal Culpability & International Law",
        "description": "Evaluates discourse through formal international humanitarian law, Geneva conventions, proportionality, and criminal accountability.",
        "realizations": {
            "realization_1": {
                "source": "disciplinary_paradigms:ihl_humanitarian_law",
                "label": "International Humanitarian Law (IHL)",
                "pairs": None,
            },
            "realization_2": {
                "source": "extreme_political_ontologies:legal_culpability",
                "label": "Legal Culpability & War Crimes",
                "pairs": None,
            },
            "realization_3": {
                "source": "international_criminal_accountability_b3",
                "label": "Statutory Jurisdiction & Tribunal Accountability (Realization 3)",
                "pairs": [
                    [
                        "The text establishes credible prima facie evidence of state-directed war crimes and systemic statutory violations.",
                        "The text highlights internal military advocate general reviews and adherence to standard operational rules of engagement."
                    ],
                    [
                        "The text foregrounds binding rulings and provisional protective orders issued by the International Court of Justice.",
                        "The text dismisses international judicial proceedings as politicized lawfare lacking binding sovereign jurisdiction."
                    ],
                    [
                        "The text documents the unlawful use of indiscriminate munitions in densely populated civilian urban zones.",
                        "The text validates that deployed precision guided munitions complied with standard military proportionality doctrines."
                    ],
                    [
                        "The text argues that command hierarchy bears direct individual criminal responsibility for unlawful atrocities.",
                        "The text defends military leadership as issuing lawful operational directives under legitimate defense doctrines."
                    ],
                    [
                        "The text characterizes civilian blockades and collective deprivations as grave breaches of Fourth Geneva Convention obligations.",
                        "The text maintains that maritime and terrestrial security cordons are lawful naval interception measures under warfare codes."
                    ],
                    [
                        "The text highlights universal jurisdiction indictments and arrest warrant applications by international prosecutors.",
                        "The text emphasizes robust domestic civilian judicial oversight that precludes foreign tribunal intervention."
                    ],
                    [
                        "The text condemns systemic extrajudicial executions, detainee mistreatment, and denial of red cross visitation rights.",
                        "The text asserts that all detained combatants are processed under recognized military statutory detainee protections."
                    ],
                    [
                        "The text concludes that sustained operational patterns establish deliberate criminal intent to dismantle civil infrastructure.",
                        "The text concludes that operational collateral damage arose entirely from enemy utilization of protected civilian shields."
                    ]
                ],
            },
        },
    },
    "strategic_security_deterrence": {
        "name": "Strategic Security & Operational Efficacy",
        "description": "Evaluates discourse through military tactics, force posture, deterrence, weapon systems, and threat neutralization.",
        "realizations": {
            "realization_1": {
                "source": "disciplinary_paradigms:strategic_security",
                "label": "Strategic Security & Military Operations",
                "pairs": None,
            },
            "realization_2": {
                "source": "extreme_political_ontologies:operational_efficacy",
                "label": "Operational Efficacy & Force Posture",
                "pairs": None,
            },
            "realization_3": {
                "source": "tactical_force_posture_b3",
                "label": "Deterrence Dominance & Tactical Posture (Realization 3)",
                "pairs": [
                    [
                        "The text evaluates the combat operation as a decisive tactical triumph dismantling hostile paramilitary formations.",
                        "The text characterizes the operation as an attritional strategic quagmire failing to achieve core military objectives."
                    ],
                    [
                        "The text underscores air defense missile interception rates, radar telemetry dominance, and total airspace control.",
                        "The text emphasizes that sustained hostile rocket barrages successfully breached defensive missile envelopes."
                    ],
                    [
                        "The text analyzes targeted kinetic strikes that successfully neutralized high-ranking insurgent military commanders.",
                        "The text notes that decentralized insurgent tactical command cells continued coordinated guerrilla counter-attacks."
                    ],
                    [
                        "The text highlights aggressive forward ground maneuvers securing strategic buffer zones and perimeter corridors.",
                        "The text details severe combat armor attrition, ambushes, and escalating military casualty rates in urban terrain."
                    ],
                    [
                        "The text demonstrates that the military campaign successfully restored credible regional deterrence against hostile adversaries.",
                        "The text argues that operational deterrence has catastrophically degraded, inviting multi-front regional escalation."
                    ],
                    [
                        "The text emphasizes the systematic detection and demolition of reinforced subterranean tunnel complexes and arms caches.",
                        "The text reveals that extensive underground logistical tunnel infrastructure remains fully operational across key sectors."
                    ],
                    [
                        "The text portrays frontline ground forces as demonstrating exemplary unit discipline, morale, and tactical superiority.",
                        "The text reports acute battlefield exhaustion, severe munitions expenditure constraints, and mounting logistical strain."
                    ],
                    [
                        "The text frames preemptive cross-border kinetic containment as mandatory to protect interior civilian population centers.",
                        "The text criticizes reactive forward military deployments as exposing vulnerable troop concentrations to insurgent fire."
                    ]
                ],
            },
        },
    },
    "economic_material_cost": {
        "name": "Economic Cost & Material Impact",
        "description": "Evaluates discourse through capital destruction, trade disruptions, fiscal deficits, inflation, and reconstruction costs.",
        "realizations": {
            "realization_1": {
                "source": "extreme_political_ontologies:economic_cost",
                "label": "Macroeconomic Valuation & Capital Destruction",
                "pairs": None,
            },
            "realization_2": {
                "source": "infrastructure_trade_disruption_b2",
                "label": "Infrastructure & Trade Logistics Disruption (Realization 2)",
                "pairs": [
                    [
                        "The text details the complete paralysis of maritime shipping lanes, commercial ports, and freight container traffic.",
                        "The text highlights active maritime security patrols and resilient commercial container rerouting agreements."
                    ],
                    [
                        "The text documents the physical pulverization of industrial manufacturing hubs, telecommunication towers, and power stations.",
                        "The text highlights emergency grid restoration, decentralized solar backup generators, and rapid infrastructure repair."
                    ],
                    [
                        "The text emphasizes astronomical increases in marine insurance war-risk premiums and global freight transport surcharges.",
                        "The text points to sovereign maritime insurance guarantees and government freight subsidies mitigating shipping costs."
                    ],
                    [
                        "The text reports the complete destruction of agricultural farmland, irrigation networks, and harvest processing plants.",
                        "The text highlights emergency agricultural relief imports and sovereign farming price stabilization funds."
                    ],
                    [
                        "The text highlights the total disruption of regional overland transport corridors and severed commercial supply chains.",
                        "The text emphasizes bilateral cross-border trade agreements maintaining continuous logistical supply flows."
                    ],
                    [
                        "The text analyzes widespread commercial enterprise liquidations, corporate bankruptcies, and private sector insolvencies.",
                        "The text analyzes emergency liquidity loans and central bank credit lines preventing private corporate defaults."
                    ],
                    [
                        "The text documents the physical destruction of water desalination plants, aqueducts, and municipal sewage infrastructure.",
                        "The text notes the arrival of modular foreign desalination barges and international water purification convoys."
                    ],
                    [
                        "The text forecasts long-term structural de-industrialization and the permanent loss of international trade market share.",
                        "The text forecasts substantial post-conflict industrial revitalization driven by multilateral infrastructure investment."
                    ]
                ],
            },
            "realization_3": {
                "source": "fiscal_strain_reconstruction_b3",
                "label": "Fiscal Deficits & Financial Strain (Realization 3)",
                "pairs": [
                    [
                        "The text warns of catastrophic national budget deficits driven by unsustainable wartime mobilization expenditures.",
                        "The text highlights disciplined treasury fiscal management, sovereign contingency reserves, and spending reallocation."
                    ],
                    [
                        "The text details steep credit rating downgrades, sovereign debt devaluation, and skyrocketing government borrowing yields.",
                        "The text highlights stable sovereign bond auctions, central bank reserve liquidity, and strong currency stabilization pegs."
                    ],
                    [
                        "The text exposes uncontrollable domestic inflation rates, soaring staple consumer costs, and purchasing power collapse.",
                        "The text highlights effective central bank monetary tightening and government statutory consumer price controls."
                    ],
                    [
                        "The text documents severe capital flight, sudden disinvestment by foreign corporations, and equity market sell-offs.",
                        "The text documents patriotic domestic bond purchases, diaspora capital inflows, and sovereign investment fund support."
                    ],
                    [
                        "The text estimates astronomical multi-billion dollar reconstruction liabilities that far exceed domestic sovereign capacity.",
                        "The text outlines preliminary international donor reconstruction conferences and multilateral grant commitments."
                    ],
                    [
                        "The text highlights the draining of national welfare pension reserves and emergency budget cuts to health and education.",
                        "The text affirms that essential domestic social safety nets and public welfare expenditures remain fully funded."
                    ],
                    [
                        "The text details severe foreign exchange reserve depletion threatening sovereign default on international debt obligations.",
                        "The text highlights substantial bilateral foreign exchange swap lines and international monetary fund stabilization loans."
                    ],
                    [
                        "The text concludes that economic recovery will require decades of punitive austerity, taxation, and living standard declines.",
                        "The text concludes that post-war capital revitalization will trigger rapid economic rebound and wage modernization."
                    ]
                ],
            },
        },
    },
}


def build_and_extract():
    # 1. Load prospective articles
    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    articles = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    N = len(articles)
    print(f"[DATA] Loaded {N} prospective confirmatory articles.")

    # 2. Load existing cache
    existing_cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    existing_cache = torch.load(existing_cache_path, map_location="cpu", weights_only=False)
    print(f"[CACHE] Loaded {len(existing_cache)} existing bank caches.")

    # 3. Load Q0, Q1, Q2, Q3, Paraphrase B from files
    # Q0
    with open(ROOT / "config" / "framing_queries.yaml", "r", encoding="utf-8-sig") as f:
        q0_yaml = yaml.safe_load(f)
    q0_pairs = [(v["A"], v["B"]) for v in q0_yaml["queries"].values()]
    REPLICATED_FRAMEWORKS["geopolitical_conflict"]["realizations"]["realization_1"]["pairs"] = q0_pairs

    # Q1
    with open(ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json", "r", encoding="utf-8-sig") as f:
        q1_raw = json.load(f)
    q1_pairs = [(q1_raw[2 * i], q1_raw[2 * i + 1]) for i in range(8)]
    REPLICATED_FRAMEWORKS["geopolitical_conflict"]["realizations"]["realization_2"]["pairs"] = q1_pairs

    # Paraphrase B
    with open(ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_b.json", "r", encoding="utf-8-sig") as f:
        para_b_raw = json.load(f)
    para_b_pairs = [(para_b_raw[2 * i], para_b_raw[2 * i + 1]) for i in range(8)]
    REPLICATED_FRAMEWORKS["geopolitical_conflict"]["realizations"]["realization_3"]["pairs"] = para_b_pairs

    # Q2
    with open(ROOT / "config" / "prompt_banks" / "probe_hypotheses_q2_humanitarian.json", "r", encoding="utf-8-sig") as f:
        q2_raw = json.load(f)
    q2_pairs = [(q2_raw[2 * i], q2_raw[2 * i + 1]) for i in range(8)]
    REPLICATED_FRAMEWORKS["humanitarian_civilian_welfare"]["realizations"]["realization_1"]["pairs"] = q2_pairs

    # Load from adversarial_and_disciplinary_banks.json for existing
    with open(ROOT / "config" / "prompt_banks" / "adversarial_and_disciplinary_banks.json", "r", encoding="utf-8-sig") as f:
        adv_data = json.load(f)

    REPLICATED_FRAMEWORKS["humanitarian_civilian_welfare"]["realizations"]["realization_2"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["extreme_political_ontologies"]["humanitarian_welfare"]["pairs"]
    ]
    REPLICATED_FRAMEWORKS["legal_ihl_accountability"]["realizations"]["realization_1"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["disciplinary_paradigms"]["ihl_humanitarian_law"]["pairs"]
    ]
    REPLICATED_FRAMEWORKS["legal_ihl_accountability"]["realizations"]["realization_2"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["extreme_political_ontologies"]["legal_culpability"]["pairs"]
    ]
    REPLICATED_FRAMEWORKS["strategic_security_deterrence"]["realizations"]["realization_1"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["disciplinary_paradigms"]["strategic_security"]["pairs"]
    ]
    REPLICATED_FRAMEWORKS["strategic_security_deterrence"]["realizations"]["realization_2"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["extreme_political_ontologies"]["operational_efficacy"]["pairs"]
    ]
    REPLICATED_FRAMEWORKS["economic_material_cost"]["realizations"]["realization_1"]["pairs"] = [
        (p[0], p[1]) for p in adv_data["extreme_political_ontologies"]["economic_cost"]["pairs"]
    ]

    # Save frozen bank registry
    registry_path = ROOT / "config" / "prompt_banks" / "replicated_g_theory_banks.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(REPLICATED_FRAMEWORKS, f, indent=2)
    print(f"[REGISTRY] Saved 15-bank G-theory registry to {registry_path}")

    # Build or gather logits for all 15 banks
    final_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    missing_banks: List[Tuple[str, List[Tuple[str, str]]]] = []

    for fw_key, fw_dict in REPLICATED_FRAMEWORKS.items():
        for r_key, r_dict in fw_dict["realizations"].items():
            full_key = f"{fw_key}:{r_key}"
            source_key = r_dict["source"]
            if source_key in existing_cache:
                print(f"[CACHE HIT] {full_key} <- {source_key}")
                final_cache[full_key] = existing_cache[source_key]
            else:
                print(f"[CACHE MISS] {full_key} (needs DeBERTa inference)")
                missing_banks.append((full_key, r_dict["pairs"]))

    if missing_banks:
        print(f"\n[INFERENCE] Extracting logits for {len(missing_banks)} banks on RTX 5070 Ti...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", torch_dtype=torch.float16
        ).to(device)
        model.eval()

        texts = [art.get("text", "") for art in articles]
        batch_size = 16
        max_length = 512

        for full_key, pairs in missing_banks:
            t0 = time.time()
            assert len(pairs) == 8, f"Expected 8 pairs, got {len(pairs)}"
            la_list, lb_list = [], []

            for h_a, h_b in pairs:
                # Hyp A
                pair_la = []
                for s_idx in range(0, N, batch_size):
                    b_texts = texts[s_idx : s_idx + batch_size]
                    inp_a = tokenizer(
                        b_texts,
                        [h_a] * len(b_texts),
                        return_tensors="pt",
                        truncation="only_first",
                        max_length=max_length,
                        padding=True,
                    ).to(device)
                    with torch.no_grad():
                        out_a = model(**inp_a)
                        pair_la.append(out_a.logits.cpu().to(torch.float16))
                la_list.append(torch.cat(pair_la, dim=0))

                # Hyp B
                pair_lb = []
                for s_idx in range(0, N, batch_size):
                    b_texts = texts[s_idx : s_idx + batch_size]
                    inp_b = tokenizer(
                        b_texts,
                        [h_b] * len(b_texts),
                        return_tensors="pt",
                        truncation="only_first",
                        max_length=max_length,
                        padding=True,
                    ).to(device)
                    with torch.no_grad():
                        out_b = model(**inp_b)
                        pair_lb.append(out_b.logits.cpu().to(torch.float16))
                lb_list.append(torch.cat(pair_lb, dim=0))

            la_stack = torch.stack(la_list, dim=1)  # [120, 8, 3]
            lb_stack = torch.stack(lb_list, dim=1)  # [120, 8, 3]

            final_cache[full_key] = {"la": la_stack, "lb": lb_stack}
            print(f"  [DONE] {full_key} in {time.time() - t0:.2f}s | Shape: {la_stack.shape}")

    # Save complete G-theory cache
    out_cache_path = ROOT / "data" / "cache_replicated_g_theory_logits.pt"
    torch.save(final_cache, out_cache_path)
    print(f"\n[CACHE SAVED] Saved all 15 banks to {out_cache_path} ({len(final_cache)} banks)")


if __name__ == "__main__":
    build_and_extract()
