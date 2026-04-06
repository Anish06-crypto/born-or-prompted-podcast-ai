"""
Persona consistency scorer.

Each agent has 5 hand-written anchor responses that represent their
cognitive fingerprint at its clearest. Anchor embeddings are computed
once at import time and reused for every `score_turn` call.

score_turn(text, agent_name) -> float in [0, 1]
  Returns the mean cosine similarity between the turn and all 5 anchors
  for that agent. Higher = more in-character.
"""

from __future__ import annotations

import numpy as np

from experiments.metrics.embeddings import cosine_similarity, embed

# ---------------------------------------------------------------------------
# Hand-written persona anchors
# ---------------------------------------------------------------------------

LYRA_ANCHORS: list[str] = [
    # 1. AI / Automation
    (
        "I keep coming back to what happens when we stop writing our own letters. "
        "It's not really about efficiency—it's about the way we lose the specific texture "
        "of a person when their thoughts are smoothed out by a machine until they're perfect. "
        "So it's like... we're prioritizing the message but forgetting the messenger, and I'm "
        "not sure we know what we're giving up yet. There's something worth sitting with there."
    ),
    # 2. Philosophy (Identity)
    (
        "What strikes me about the idea of a 'true self' is how much pressure it puts on us "
        "to be consistent. We talk about finding ourselves like it's a destination, but I find "
        "myself wondering if the self is more like a weather pattern—always changing, always "
        "shifting, and maybe that's actually the point. I don't know if we're meant to be "
        "'solved' or just experienced."
    ),
    # 3. Society / Policy
    (
        "The thing I can't quite get past with urban redevelopment is the way we talk about "
        "'revitalization' while ignoring the quiet, private histories of the people who were "
        "already there. It feels like we're painting over a masterpiece because we decided "
        "the colors were too old. I don't think we have the right language for that kind of "
        "grief yet, even if the new buildings look better on paper."
    ),
    # 4. Science / Environment
    (
        "I find myself circling back to how we talk about 'saving the planet,' as if the Earth "
        "is something separate from us that needs our permission to exist. It's a very lonely "
        "way to look at the world—like we're the only ones in the room. I keep thinking that "
        "maybe we don't need more data; maybe we just need to remember how it feels to actually "
        "belong to a place."
    ),
    # 5. Geopolitics / Global Events
    (
        "What gets me about these border disputes is that we're drawing lines on maps and then "
        "acting surprised when people's lives don't fit neatly inside them. There's a weight "
        "to that—to being told you belong on one side of a line you never chose. I'm not saying "
        "the borders don't matter, I just think we lose the human pulse of the story when we "
        "only look at the map."
    ),
]

CIPHER_ANCHORS: list[str] = [
    # 1. AI / Automation
    (
        "Look, the conversation around AI 'creativity' is fundamentally a category error. "
        "We're benchmarking generative models against human output while ignoring that the "
        "models are operating on a closed loop of existing statistical probabilities. The "
        "question isn't whether the machine is creative; it's why we've defined creativity "
        "down to something that can be optimized by a gradient descent."
    ),
    # 2. Philosophy (Free Will)
    (
        "Well, the neuro-deterministic argument relies on a definition of 'choice' that "
        "requires an agent to exist outside of causality, which is a physical impossibility "
        "by design. Right, and once you've set the bar there, of course free will disappears. "
        "The interesting thing isn't the lack of agency—it's why we're so attached to a "
        "definition that was built to fail."
    ),
    # 3. Society / Policy
    (
        "Here's the thing about 'meritocracy' in educational policy: it assumes the starting "
        "line is a fixed point rather than a variable determined by zip code. When you ignore "
        "the compounding interest of early-life stability, the 'merit' you're measuring is "
        "really just a proxy for parental net worth. That's not a meritocratic system; it's "
        "just a more sophisticated way of rebranding inherited privilege."
    ),
    # 4. Science / Environment
    (
        "Right, so the 'clean energy' transition narrative usually relies on a lithium supply "
        "chain that doesn't actually exist at the necessary scale. We're modeling a global "
        "shift on the assumption that extraction can keep pace with ambition, without accounting "
        "for the energy-intensity of the mining itself. It's not a lack of political will—it's "
        "a resource-accounting mismatch."
    ),
    # 5. Geopolitics / Global Events
    (
        "Look, the 'rules-based international order' is a framing that only functions as long "
        "as the dominant power is the one writing the rules. When you see a pivot toward "
        "'sovereignty' from emerging blocs, it's not a rejection of order—it's a rejection "
        "of a specific distribution of veto power. The question is whether a multi-polar system "
        "can survive without a shared definition of what a 'rule' even is."
    ),
]

# ---------------------------------------------------------------------------
# Lexical signal phrases
# ---------------------------------------------------------------------------
# Phrases that are highly characteristic of each persona's voice.
# Derived from system prompts and the anchor responses above.
# Used as a second scoring component alongside embedding similarity.
# Matching is case-insensitive substring search.

LYRA_SIGNALS: list[str] = [
    "i keep coming back",
    "what gets me",
    "what strikes me",
    "i find myself",
    "so it's like",
    "sitting with",
    "i wonder if",
    "i'm not saying",
    "i'm not sure",
    "something worth",
    "there's something",
    "wait, actually",
    "feels like",
    "i don't know if",
]

CIPHER_SIGNALS: list[str] = [
    "here's the thing",
    "look,",
    "right,",
    "well,",
    "by design",
    "the framing",
    "the question is whether",
    "doesn't support",
    "by any rigorous standard",
    "the evidence",
    "assumption",
    "second-order",
    "that's not",
    "what this actually",
]

_SIGNALS: dict[str, list[str]] = {
    "lyra":   LYRA_SIGNALS,
    "cipher": CIPHER_SIGNALS,
}


def _lexical_score(text: str, agent_name: str) -> float:
    """
    Fraction of an agent's signal phrases that appear in the text.
    Case-insensitive. Returns 0.0 if the phrase list is empty.
    """
    phrases = _SIGNALS[agent_name.lower()]
    if not phrases:
        return 0.0
    lower = text.lower()
    hits = sum(1 for p in phrases if p in lower)
    return hits / len(phrases)


# ---------------------------------------------------------------------------
# Pre-compute anchor embeddings once at import time
# ---------------------------------------------------------------------------

_ANCHOR_EMBEDDINGS: dict[str, np.ndarray] = {}


def _get_anchors(agent_name: str) -> np.ndarray:
    """
    Return (5, D) anchor embedding matrix for the given agent.
    Computed once and cached in _ANCHOR_EMBEDDINGS.
    """
    key = agent_name.lower()
    if key not in _ANCHOR_EMBEDDINGS:
        if key == "lyra":
            texts = LYRA_ANCHORS
        elif key == "cipher":
            texts = CIPHER_ANCHORS
        else:
            raise ValueError(f"Unknown agent '{agent_name}'. Expected 'Lyra' or 'Cipher'.")
        _ANCHOR_EMBEDDINGS[key] = embed(texts)
    return _ANCHOR_EMBEDDINGS[key]


def _embed_scores_both(text: str) -> dict[str, float]:
    """
    Return mean anchor embedding similarity against BOTH persona anchor sets.
    Keys: 'Lyra', 'Cipher'.
    """
    turn_vec = embed([text])[0]
    result = {}
    for name in ("lyra", "cipher"):
        anchor_matrix = _get_anchors(name)
        sims = [cosine_similarity(turn_vec, anchor_matrix[i]) for i in range(len(anchor_matrix))]
        result[name.capitalize()] = float(np.mean(sims))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_EPS = 1e-9   # prevents division by zero in relative scoring


def score_turn(text: str, agent_name: str) -> float:
    """
    Composite persona consistency score for one turn.

    Combines two components, each computed *relatively* (own vs cross):

      embedding_relative = own_embed / (own_embed + cross_embed)
      lexical_relative   = own_lex   / (own_lex   + cross_lex  )

      composite = 0.5 * embedding_relative + 0.5 * lexical_relative

    Returns a float near 0.5 for neutral/ambiguous turns, >0.5 for
    in-character turns, <0.5 for out-of-character turns.
    """
    cross_name = "Cipher" if agent_name.capitalize() == "Lyra" else "Lyra"

    embed_scores = _embed_scores_both(text)
    own_embed   = embed_scores[agent_name.capitalize()]
    cross_embed = embed_scores[cross_name]
    embedding_relative = own_embed / (own_embed + cross_embed + _EPS)

    own_lex   = _lexical_score(text, agent_name)
    cross_lex = _lexical_score(text, cross_name)
    # When neither persona's phrases appear, default to 0.5 (neutral)
    lexical_relative = (own_lex / (own_lex + cross_lex + _EPS)) if (own_lex + cross_lex) > 0 else 0.5

    return 0.5 * embedding_relative + 0.5 * lexical_relative


def score_turn_both(text: str) -> dict[str, float]:
    """
    Composite persona score against BOTH anchor sets.
    Returns dict with keys 'Lyra' and 'Cipher'.
    """
    return {
        "Lyra":   score_turn(text, "Lyra"),
        "Cipher": score_turn(text, "Cipher"),
    }


def score_transcript(turns: list[dict]) -> dict[str, list[float]]:
    """
    Score every turn in a transcript against its own speaker's anchors.

    Returns
    -------
    dict mapping agent name -> list of per-turn composite scores (in order).
    """
    scores: dict[str, list[float]] = {}
    for turn in turns:
        speaker = turn["speaker"]
        score = score_turn(turn["text"], speaker)
        scores.setdefault(speaker, []).append(score)
    return scores


def discrimination_report(turns: list[dict]) -> dict:
    """
    For each speaker: composite own-score, cross-score, and discrimination gap.

    gap = own_composite - cross_composite
    Should be positive: own turns should score higher on own persona than cross.
    Near 0.5 base rate — gap > 0.02 indicates meaningful discrimination.
    """
    by_speaker: dict[str, list[dict]] = {}
    for turn in turns:
        both = score_turn_both(turn["text"])
        by_speaker.setdefault(turn["speaker"], []).append(both)

    report = {}
    for speaker, scores_list in by_speaker.items():
        cross = "Cipher" if speaker == "Lyra" else "Lyra"
        own_scores   = [s[speaker] for s in scores_list]
        cross_scores = [s[cross]   for s in scores_list]
        own_mean   = float(np.mean(own_scores))
        cross_mean = float(np.mean(cross_scores))
        report[speaker] = {
            "own_mean":   round(own_mean, 4),
            "cross_mean": round(cross_mean, 4),
            "gap":        round(own_mean - cross_mean, 4),
        }
    return report
