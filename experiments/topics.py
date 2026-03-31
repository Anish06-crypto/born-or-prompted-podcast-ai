"""
Standardized topic bank for controlled experiments.

Eight topics across six distinct domains. Selected criteria:
  - Both sides have genuine arguments (elicits real debate)
  - No domain appears more than twice (controls for domain familiarity bias)
  - Covers emotional, analytical, policy, and scientific registers
  - No clear "correct" answer that would cause agents to converge rather than debate

DO NOT modify these topics once a study run has started.
Topic indices (0–7) are baked into condition IDs and must remain stable.
"""

TOPICS: list[str] = [
    # 0 — Technology / Labour
    "AI will eliminate more jobs than it creates within the next decade",

    # 1 — Society / Psychology
    "Social media has done more harm than good to mental health and democracy",

    # 2 — Economics / Policy
    "Universal basic income would reduce workforce participation and innovation over time",

    # 3 — Science / Ethics
    "The resources spent on space exploration would produce greater good if redirected to Earth's problems",

    # 4 — Governance / Technology
    "Strong government regulation of AI development will cause more harm than good",

    # 5 — Bioethics
    "Genetic editing to prevent hereditary diseases in human embryos should be legally permitted",

    # 6 — Work / Society
    "Remote work has weakened organisational culture more than it has empowered individuals",

    # 7 — Environment / Economics
    "Meaningful climate action and sustained economic growth are fundamentally incompatible",
]
