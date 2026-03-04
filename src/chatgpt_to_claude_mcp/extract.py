#!/usr/bin/env python3
"""
Pipeline:
  1. Extract user-only messages from scraped markdown files
  2. Build word/phrase frequency map
  3. Grep for self-referential signal phrases
  4. Score signal lines by frequency relevance + Shannon entropy,
     output a dense_signal.txt trimmed to TARGET_CHARS
"""

import re
import math
import json
import sys
from pathlib import Path
from collections import Counter

TOP_N = 500  # top words/bigrams to show in frequency map
TARGET_CHARS = 50_000   # target size for dense_signal.txt
FREQ_WEIGHT    = 0.60   # weight for frequency relevance score
ENTROPY_WEIGHT = 0.40   # weight for Shannon entropy score
BANNED_PENALTY = 0.05   # score multiplier for lines containing banned phrases
CONTEXT_MAX    = 3      # max surrounding lines to add per high-scoring line

# Phrases that should be heavily penalised — loaded from banned_phrases.json at runtime
BANNED_PHRASES: set[str] = set()

SIGNAL_PHRASES = [
    "i ", "i'm", "i've", "i'd", "i'll",
    "my ", "we're", "we are",
    "i decided", "i chose", "i picked",
    "i hate", "i love", "i prefer", "i like", "i want", "i need",
    "i always", "i never",
    "always", "never",
    "insane", "the problem", "okay so", "okay it's",
]

STOPWORDS = {
    # common English
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "that","this","it","is","was","are","were","be","been","have","has","had",
    "do","did","does","not","no","so","if","as","by","from","up","about","into",
    "than","then","they","their","them","there","its","you","your","we","our",
    "just","like","get","got","can","will","would","could","should","also",
    "very","more","some","what","when","how","all","one","out","he","she",
    "which","who","him","her","his","me","my","i","yeah","ok","okay",
    # pasted log/debug output
    "warning","logtemp","logblueprintusermessages","info","trace","sending",
    # physics variable names (tire model / vehicle sim code)
    "tire","vvert","vlong","vlat","normalforce","slopescale","staticload",
    "effmass","deltaratio","smoothdelta","vertforce","latforce","latmagnitude",
    "slipratio","forwardalpha","wheelomega","wheelvelocity","longpacejka",
    "gravityslope","longmagnitude","longforce","ellipsescale","totalforce",
    "frontdrift","reardrift","rearaccel","frontaccel","steerang","dotproduct",
    "zerovector","gripsettings","gripx","gripy","latload","slipangle","slipang",
    "springsettings","restlen","slopefactor","leverarm","yawtorque","tireforce",
    "updatetireforces","forwardspeedmod","latmagnitude","ellipse","stiff",
    "fade","slope","atan","sin","rad","deg","curved","shaped","fwd","peak",
    "compression","float","const","fvector",
    # code / file tokens
    "src","bin","bytes","webp","png","jpg","gif","jsx","dom","export","return",
    "https","www","com","http",
    # LSP / editor noise
    "textdocument","didopen","notification","sediment",
    # usernames / handles
    "silentfactory","chrisconiglio","nvizzio","tencircles","feelslike","gweb",
    # UI / frontend code tokens
    "controller","assets","images","artboards","app","mode","public",
    "components","pages","dom","jsx","blockid","spaceid","craftdocs","repos",
    # physics / math (remaining)
    "mag","torque","inputs","params","longitudinal","lateral","slip","velocity",
    "alpha","delta","lat","fmath","abs","restlength","springlength","wheelradius",
    "fwdmod","slipin","rawmag","spring","length","shape","curve","raw",
    # devops / system noise
    "ssh","cpu","throw","logaudiomixer","display","designentry","bmagnet","binair",
    "configuration","engine","git","push","pull","commit","branch","clone","merge",
    # remaining physics interpolation variables
    "deltainterpspeed","smoothinterpspeed","restlength","springlength",
}

# Bigrams to suppress even if individual words pass the stopword filter
BIGRAM_STOPWORDS = {
    "chris coniglio","silent factory","let know","forward right",
    "sub steps","speed speed","cloud demo","right now","don know",
    "mag mag","repos cloud","avatar ago","chris chris",
    "subject verb","verb object","noun enemy","noun weapon","noun cosmetic",
    "noun location","noun effect","kind name","alpha delta","slip angle",
    "file service","service file","right drift","drift forward",
    "right mag","total drift","don think","don want","don need","don really",
    "build started","ago looks","best chris","hey chris",
    "logaudiomixer display","steps designentry","throw new",
    "configuration development","development any","any cpu","engine source",
    "users ssh","context mode","mode context","state context",
    "binair bmagnet","raw params","params inputs","inputs velocity",
    # remaining physics / generic noise bigrams
    "right right","speed result","result total","drift final","final drift",
    "create next","speed speed","drift speed",
}


# ── Step 1: Extract user messages ─────────────────────────────────────────────

def extract_user_messages(md_path: Path) -> list[str]:
    """Return a list of user message blocks from a markdown conversation file."""
    messages = []
    current = []
    capturing = False

    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("**You**"):
            capturing = True
            current = []
        elif line.startswith("**Assistant**"):
            if capturing and current:
                messages.append(" ".join(current).strip())
            capturing = False
            current = []
        elif line == "---":
            if capturing and current:
                messages.append(" ".join(current).strip())
            capturing = False
            current = []
        elif capturing:
            stripped = line.strip()
            if stripped:
                current.append(stripped)

    if capturing and current:
        messages.append(" ".join(current).strip())

    return messages


# ── Step 2: Frequency map ──────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())

def build_frequency(all_text: str, top_n: int):
    tokens = tokenize(all_text)
    words  = [t for t in tokens if t not in STOPWORDS]
    bigrams = [
        f"{words[i]} {words[i+1]}"
        for i in range(len(words)-1)
        if f"{words[i]} {words[i+1]}" not in BIGRAM_STOPWORDS
    ]

    word_freq   = Counter(words).most_common(top_n)
    bigram_freq = Counter(bigrams).most_common(top_n)
    return word_freq, bigram_freq


# ── Step 3: Signal grep ────────────────────────────────────────────────────────

def find_signal_lines(all_lines: list[str]) -> list[tuple[int, str]]:
    """Return (line_index, line) for lines matching signal phrases, minus artifacts."""
    artifact_re = re.compile(
        r"file-service://|sediment://|image_asset_pointer|audio_asset_pointer"
        r"|audio_transcription|real_time_user|asset_pointer"
        r"|private key|git |\.git|ssh-|BEGIN RSA|BEGIN EC"
        r"|nvizzio|sylvain|constantin|yves|gamevestor|ivan"
    )
    results = []
    for idx, line in enumerate(all_lines):
        if artifact_re.search(line):
            continue
        lower = line.lower()
        if any(phrase in lower for phrase in SIGNAL_PHRASES):
            results.append((idx, line.strip()))
    return results


# ── Step 4: Score signal lines ────────────────────────────────────────────────

def shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def score_signal_lines(signal_lines: list[tuple[int, str]], top_words: set[str],
                       all_lines: list[str], target_chars: int) -> list[str]:
    """
    Score each signal line by a weighted sum of:
      - frequency score: fraction of tokens that are top-frequency words
      - entropy score:   normalised Shannon entropy
    Apply BANNED_PENALTY multiplier to lines containing banned phrases.
    Expand each selected line with ±randint(0, CONTEXT_MAX) surrounding lines.
    """
    import random
    MAX_ENTROPY = 4.5

    scored = []
    for idx, line in signal_lines:
        tokens = re.findall(r"\b[a-z]{3,}\b", line.lower())
        if not tokens:
            continue
        freq_score = sum(1 for t in tokens if t in top_words) / len(tokens)
        ent_score  = min(shannon_entropy(line) / MAX_ENTROPY, 1.0)
        score = FREQ_WEIGHT * freq_score + ENTROPY_WEIGHT * ent_score

        # Penalise lines containing banned phrases
        lower = line.lower()
        if any(bp in lower for bp in BANNED_PHRASES):
            score *= BANNED_PENALTY

        scored.append((score, idx, line))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Expand with surrounding context, deduplicate, trim to target_chars
    seen_indices: set[int] = set()
    result, total = [], 0

    for _, idx, line in scored:
        if total >= target_chars:
            break

        # Gather this line + random surrounding context
        radius = random.randint(0, CONTEXT_MAX)
        context_indices = range(max(0, idx - radius),
                                min(len(all_lines), idx + radius + 1))
        for ci in context_indices:
            if ci in seen_indices:
                continue
            ctx_line = all_lines[ci].strip()
            if not ctx_line:
                continue
            if any(bp in ctx_line.lower() for bp in BANNED_PHRASES):
                continue
            if total + len(ctx_line) + 1 > target_chars:
                break
            result.append(ctx_line)
            seen_indices.add(ci)
            total += len(ctx_line) + 1

    return result




def load_banned_phrases(export_dir: Path) -> set[str]:
    """Load banned phrases from export_dir/banned_phrases.json."""
    config_path = export_dir / "banned_phrases.json"
    if config_path.exists():
        return set(json.loads(config_path.read_text()))
    return set()


def save_banned_phrases(export_dir: Path, phrases: set[str]) -> None:
    config_path = export_dir / "banned_phrases.json"
    config_path.write_text(json.dumps(sorted(phrases), indent=2))


def run_extraction(input_dir: Path, output_dir: Path,
                   banned: set[str] | None = None) -> dict:
    """Run the full extraction pipeline. Returns stats + paths to output files."""
    global BANNED_PHRASES
    BANNED_PHRASES = banned or set()

    output_dir.mkdir(exist_ok=True)
    md_files = sorted(input_dir.glob("*.md"))

    all_user_text_parts = []
    for md_file in md_files:
        all_user_text_parts.extend(extract_user_messages(md_file))

    all_user_text = "\n".join(all_user_text_parts)
    all_lines = all_user_text.splitlines()

    (output_dir / "user_messages.txt").write_text(all_user_text, encoding="utf-8")

    word_freq, bigram_freq = build_frequency(all_user_text, TOP_N)
    freq_lines = [f"TOP {TOP_N} WORDS", "=" * 40]
    for word, count in word_freq:
        freq_lines.append(f"  {count:>6}  {word}")
    freq_lines += ["", f"TOP {TOP_N} BIGRAMS", "=" * 40]
    for bigram, count in bigram_freq:
        freq_lines.append(f"  {count:>6}  {bigram}")
    (output_dir / "frequency.txt").write_text("\n".join(freq_lines), encoding="utf-8")

    signal_lines = find_signal_lines(all_lines)
    (output_dir / "signal.txt").write_text(
        "\n".join(line for _, line in signal_lines), encoding="utf-8"
    )

    top_words = {w for w, _ in word_freq}
    dense_lines = score_signal_lines(signal_lines, top_words, all_lines, TARGET_CHARS)
    dense_path = output_dir / "dense_signal.txt"
    dense_path.write_text("\n".join(dense_lines), encoding="utf-8")

    # Extract top proper nouns — capitalised words that appear mid-sentence
    # (i.e., NOT at the start of a line, to avoid sentence-initial false positives)
    all_tokens = re.findall(r"(?<=[a-z,;:]\s)[A-Z][a-z]{2,}", all_user_text)
    proper_noun_freq = Counter(
        t for t in all_tokens if t.lower() not in STOPWORDS
    ).most_common(30)

    # Top bigrams — reveal recurring topics and verb phrases
    top_bigrams = [
        {"phrase": bg, "count": c}
        for bg, c in bigram_freq[:40]
    ]

    return {
        "md_files": len(md_files),
        "user_messages": len(all_user_text_parts),
        "dense_signal_chars": dense_path.stat().st_size,
        "dense_signal_lines": len(dense_lines),
        "top_proper_nouns": [{"word": w, "count": c} for w, c in proper_noun_freq],
        "top_phrases": top_bigrams,
        "output_dir": str(output_dir),
    }


def main():
    input_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extracted")
    banned = load_banned_phrases(input_dir.parent)
    stats = run_extraction(input_dir, output_dir, banned)
    print(f"Done. {stats['user_messages']} messages → {output_dir}/")
    print(f"Top proper nouns: {[n['word'] for n in stats['top_proper_nouns'][:10]]}")


if __name__ == "__main__":
    main()
