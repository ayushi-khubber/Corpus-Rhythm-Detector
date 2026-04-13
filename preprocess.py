# preprocess.py
"""
Corpus Rhythm Detector - NLP Preprocessing Pipeline
Analyzes temporal word frequency patterns in novels.
"""

import re
import json
import math
from collections import Counter
from pathlib import Path


# ─── Configuration ─────────────────────────────────────────────────────────────

NUM_SEGMENTS = 90  # Target number of equal text segments

# Predefined word sets for each novel
WORD_SETS = {
    "moby_dick": ["whale", "ahab", "sea", "ship", "white", "captain", "water", "boat"],
    "war_and_peace": ["war", "peace", "pierre", "natasha", "napoleon", "prince", "battle", "love"],
    "crime_and_punishment": [
        "guilt", "raskolnikov", "crime", "murder", "punishment", "soul", "money", "suffer"
    ],
}

# Project Gutenberg boundary markers
GUTENBERG_START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "**START OF THE PROJECT GUTENBERG",
]
GUTENBERG_END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "**END OF THE PROJECT GUTENBERG",
]


# ─── Text Cleaning ──────────────────────────────────────────────────────────────

def remove_gutenberg_boilerplate(text: str) -> str:
    """Strip Project Gutenberg header and footer from raw text."""
    start_idx = 0
    end_idx = len(text)

    for marker in GUTENBERG_START_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Skip to end of that line
            start_idx = text.find("\n", idx) + 1
            break

    for marker in GUTENBERG_END_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx]


def clean_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"\r\n|\r", "\n", text)          # Normalize line endings
    text = re.sub(r"[ \t]+", " ", text)             # Collapse spaces/tabs
    return text.strip()


# ─── Tokenization ───────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Extract word tokens using the project regex pattern."""
    return re.findall(r"\b[a-z']+\b", text)


# ─── Segmentation ───────────────────────────────────────────────────────────────

def split_into_segments(tokens: list[str], num_segments: int = NUM_SEGMENTS):
    """
    Divide token list into equal-sized segments.
    Returns a list of (segment_tokens, start_index, end_index) tuples.
    """
    total = len(tokens)
    seg_size = math.ceil(total / num_segments)
    segments = []

    for i in range(0, total, seg_size):
        chunk = tokens[i : i + seg_size]
        if chunk:
            segments.append((chunk, i, min(i + seg_size, total)))

    return segments


# ─── Frequency Calculation ───────────────────────────────────────────────────────

def compute_normalized_frequency(tokens: list[str], word: str) -> float:
    """
    Normalized frequency = (count / total_words) * 10,000
    """
    count = tokens.count(word)
    total = len(tokens)
    if total == 0:
        return 0.0
    return round((count / total) * 10000, 4)


# ─── Pattern Detection ───────────────────────────────────────────────────────────

def detect_peaks(series: list[float], threshold_factor: float = 1.5) -> list[int]:
    """
    Detect local maxima (peaks) in a frequency series.
    A point is a peak if it exceeds threshold_factor * mean of its neighbors.
    """
    if len(series) < 3:
        return []

    mean_val = sum(series) / len(series)
    threshold = mean_val * threshold_factor
    peaks = []

    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            if series[i] > threshold:
                peaks.append(i)

    return peaks


def detect_trend(series: list[float]) -> dict:
    """
    Compute linear trend slope using least squares regression.
    Returns slope value and categorical label.
    """
    n = len(series)
    if n < 2:
        return {"slope": 0.0, "label": "flat"}

    x_mean = (n - 1) / 2
    y_mean = sum(series) / n

    numerator = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator != 0 else 0.0

    if slope > 0.005:
        label = "increasing"
    elif slope < -0.005:
        label = "decreasing"
    else:
        label = "flat"

    return {"slope": round(slope, 6), "label": label}


# ─── Pearson Correlation ─────────────────────────────────────────────────────────

def pearson_correlation(series_a: list[float], series_b: list[float]) -> float:
    """Compute Pearson correlation coefficient between two frequency series."""
    n = len(series_a)
    if n < 2 or len(series_b) != n:
        return 0.0

    mean_a = sum(series_a) / n
    mean_b = sum(series_b) / n

    numerator = sum((series_a[i] - mean_a) * (series_b[i] - mean_b) for i in range(n))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in series_a))
    denom_b = math.sqrt(sum((x - mean_b) ** 2 for x in series_b))

    if denom_a == 0 or denom_b == 0:
        return 0.0

    return round(numerator / (denom_a * denom_b), 4)


# ─── Insight Generation ──────────────────────────────────────────────────────────

def generate_insights(word: str, series: list[float], peaks: list[int], trend: dict) -> list[str]:
    """Generate human-readable literary insights from detected patterns."""
    insights = []
    mean_freq = sum(series) / len(series) if series else 0

    # Trend insight
    if trend["label"] == "increasing":
        insights.append(
            f'"{word}" shows a strong increasing trend — '
            f"its presence grows as the narrative progresses."
        )
    elif trend["label"] == "decreasing":
        insights.append(
            f'"{word}" fades across the text — '
            f"its thematic weight diminishes toward the end."
        )
    else:
        insights.append(
            f'"{word}" maintains a relatively stable presence throughout the novel.'
        )

    # Peak insight
    if len(peaks) > 5:
        insights.append(
            f'"{word}" shows {len(peaks)} periodic spikes — '
            f"suggesting rhythmic narrative recurrence."
        )
    elif len(peaks) > 0:
        peak_positions = ", ".join(str(p + 1) for p in peaks[:3])
        insights.append(
            f'"{word}" peaks at segments {peak_positions} — '
            f"likely corresponding to key narrative moments."
        )

    # Frequency insight
    if mean_freq > 20:
        insights.append(
            f'"{word}" is a dominant lexical presence '
            f"(avg {mean_freq:.1f} per 10k words)."
        )
    elif mean_freq < 1:
        insights.append(
            f'"{word}" is sparse (avg {mean_freq:.2f} per 10k words), '
            f"appearing only in concentrated bursts."
        )

    return insights


# ─── Main Pipeline ───────────────────────────────────────────────────────────────

def process_novel(
    txt_path: str,
    target_words: list[str],
    novel_id: str,
    novel_title: str,
    num_segments: int = NUM_SEGMENTS,
) -> dict:
    """
    Full preprocessing pipeline for a single novel.

    Args:
        txt_path:     Path to the .txt file
        target_words: List of words to track
        novel_id:     Short identifier (e.g., 'moby_dick')
        novel_title:  Display title
        num_segments: How many segments to divide the novel into

    Returns:
        Structured analysis dict ready for JSON serialization
    """
    print(f"\nProcessing: {novel_title}")
    print(f"  File: {txt_path}")

    # ── Load raw text ────────────────────────────────
    raw = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    print(f"  Raw characters: {len(raw):,}")

    # ── Clean ────────────────────────────────────────
    cleaned = remove_gutenberg_boilerplate(raw)
    cleaned = clean_text(cleaned)
    print(f"  Cleaned characters: {len(cleaned):,}")

    # ── Tokenize ─────────────────────────────────────
    tokens = tokenize(cleaned)
    print(f"  Total tokens: {len(tokens):,}")

    # ── Segment ──────────────────────────────────────
    segments_data = split_into_segments(tokens, num_segments)
    actual_segments = len(segments_data)
    print(f"  Segments: {actual_segments}")

    # Reconstruct original char positions for snippet extraction
    # (approximate — join tokens back per segment)
    segment_snippets = []
    for seg_tokens, _, _ in segments_data:
        snippet = " ".join(seg_tokens[:40])  # ~200 chars approx
        segment_snippets.append(snippet[:200])

    # ── Compute word frequency series ────────────────
    word_series = {}
    word_patterns = {}

    for word in target_words:
        series = [
            compute_normalized_frequency(seg_tokens, word)
            for seg_tokens, _, _ in segments_data
        ]

        peaks = detect_peaks(series)
        trend = detect_trend(series)
        insights = generate_insights(word, series, peaks, trend)

        word_series[word] = series
        word_patterns[word] = {
            "peaks": peaks,
            "trend": trend,
            "insights": insights,
            "max_frequency": round(max(series), 4),
            "mean_frequency": round(sum(series) / len(series), 4),
        }

    # ── Compute pairwise correlations ────────────────
    correlations = {}
    for i, w1 in enumerate(target_words):
        for w2 in target_words[i + 1 :]:
            key = f"{w1}_vs_{w2}"
            r = pearson_correlation(word_series[w1], word_series[w2])
            correlations[key] = r

    # ── Build output structure ────────────────────────
    output = {
        "meta": {
            "novel_id": novel_id,
            "title": novel_title,
            "total_tokens": len(tokens),
            "num_segments": actual_segments,
            "target_words": target_words,
        },
        "segments": [
            {
                "index": idx,
                "token_count": len(seg_tokens),
                "snippet": segment_snippets[idx],
            }
            for idx, (seg_tokens, _, _) in enumerate(segments_data)
        ],
        "words": word_series,
        "patterns": word_patterns,
        "correlations": correlations,
    }

    return output


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Demo mode: process all three novels if files exist ──
    novels = [
        {
            "path": "moby_dick.txt",
            "id": "moby_dick",
            "title": "Moby Dick",
            "words": WORD_SETS["moby_dick"],
        },
        {
            "path": "war_and_peace.txt",
            "id": "war_and_peace",
            "title": "War and Peace",
            "words": WORD_SETS["war_and_peace"],
        },
        {
            "path": "crime_and_punishment.txt",
            "id": "crime_and_punishment",
            "title": "Crime and Punishment",
            "words": WORD_SETS["crime_and_punishment"],
        },
    ]

    results = {}

    for novel in novels:
        path = Path(novel["path"])
        if not path.exists():
            print(f"  Skipping {novel['title']} — file not found: {novel['path']}")
            continue

        data = process_novel(
            txt_path=str(path),
            target_words=novel["words"],
            novel_id=novel["id"],
            novel_title=novel["title"],
        )
        results[novel["id"]] = data

    if not results:
        print("\nNo .txt files found. Please download novels from Project Gutenberg.")
        print("  Moby Dick:             https://www.gutenberg.org/ebooks/2701")
        print("  War and Peace:         https://www.gutenberg.org/ebooks/2600")
        print("  Crime and Punishment:  https://www.gutenberg.org/ebooks/2554")
        sys.exit(1)

    # Write combined JSON output
    out_path = Path("analysis.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nOutput written to: {out_path}")
    print(f"Novels processed: {list(results.keys())}")
