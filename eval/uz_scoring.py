"""Automated quality signals for Uzbek model output.

None of these replace a native reader, but together they catch the failure
modes that actually show up when a multilingual model is weak in Uzbek:

  1. script_drift        - asked for Latin, answered in Cyrillic (or vice versa)
  2. turkish_contamination - drifted into Turkish/Azeri orthography (ç ş ğ ı ö ü)
  3. english_leakage     - answered in English despite an Uzbek instruction
  4. russian_leakage     - Russian words/letters bleeding into Uzbek Cyrillic
  5. unseen_trigram_rate - orthographically implausible "words" (the failure mode
                           where a heavily-quantised model emits Uzbek-looking
                           non-words). Compares character trigrams against the
                           suite's reference corpus.
  6. known_word_rate     - share of tokens recognisable as Uzbek after stripping
                           agglutinative suffixes
  7. suffix_rate         - is the model actually agglutinating, or emitting
                           bare stems in Uzbek-flavoured word salad?
  8. repetition_rate     - degeneration / looping

Caveat that matters: the reference corpus and lexicon here are small. Treat
unseen_trigram_rate and known_word_rate as *comparative* signals between runs
(CPU vs GPU, quant A vs quant B), not as absolute measures of Uzbek quality.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Uzbek Latin does NOT contain these. Their presence means the model slid into
# Turkish or Azeri, which is the single most common Uzbek failure mode for
# multilingual models (the languages are close enough to attract the decoder).
TURKISH_ONLY_CHARS = set("çşğıöüÇŞĞİÖÜâî")

# Uzbek Cyrillic has no ы, ь, щ. Russian does.
RUSSIAN_ONLY_CHARS = set("ыьщЫЬЩ")
RUSSIAN_STOPWORDS = {
    "и", "в", "не", "на", "что", "это", "для", "как", "но", "по", "из", "он",
    "она", "они", "мы", "вы", "был", "была", "было", "быть", "или", "если",
    "так", "уже", "все", "его", "her", "который", "также",
}

# Uzbek's two apostrophe conventions.
MODIFIER_TURNED_COMMA = "ʻ"   # oʻ gʻ  (correct)
MODIFIER_APOSTROPHE = "ʼ"     # ʼ tutuq belgisi (correct)
ASCII_APOSTROPHES = set("'‘’`")

# Common English function words, for detecting an English answer.
ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "are", "was", "were", "that", "this",
    "with", "for", "as", "it", "on", "be", "by", "an", "or", "from", "which",
    "you", "your", "we", "they", "have", "has", "can", "will", "would", "there",
}

# ---------------------------------------------------------------------------
# Uzbek lexicon: high-frequency function words + common content words.
# Deliberately small and hand-checked; used with suffix stripping.
# ---------------------------------------------------------------------------
UZBEK_CORE_WORDS = {
    # pronouns / determiners
    "men", "sen", "u", "biz", "siz", "ular", "bu", "shu", "o'sha", "oʻsha",
    "har", "hech", "ba'zi", "baʼzi", "barcha", "hamma", "o'z", "oʻz", "kim",
    "nima", "qanday", "qancha", "qayer", "qachon", "nega", "nechta", "qaysi",
    # conjunctions / particles
    "va", "bilan", "uchun", "lekin", "ammo", "biroq", "ham", "yoki", "agar",
    "chunki", "shuning", "shunday", "esa", "emas", "yo'q", "yoʻq", "bor",
    "kerak", "mumkin", "faqat", "yana", "hali", "endi", "keyin", "oldin",
    "so'ng", "soʻng", "so'nggi", "soʻnggi", "juda", "eng", "ko'p", "koʻp",
    "oz", "kam", "balki", "albatta", "masalan", "ya'ni", "yaʼni",
    # verbs (stems and very common forms)
    "bo'l", "boʻl", "bo'ldi", "boʻldi", "bo'lgan", "boʻlgan", "bo'ladi",
    "boʻladi", "qil", "qildi", "qilgan", "qiladi", "qilish", "ber", "berdi",
    "bergan", "beradi", "kel", "keldi", "kelgan", "keladi", "bor", "bordi",
    "boradi", "ket", "ketdi", "ketadi", "ko'r", "koʻr", "ko'rdi", "koʻrdi",
    "ko'radi", "koʻradi", "de", "dedi", "deydi", "degan", "ol", "oldi",
    "oladi", "yoz", "yozdi", "yozadi", "yozish", "o'qi", "oʻqi", "o'qidi",
    "oʻqidi", "o'qiydi", "oʻqiydi", "ishla", "ishlaydi", "yasha", "yashaydi",
    "bil", "bildi", "biladi", "ayt", "aytdi", "aytadi", "top", "topdi",
    "topadi", "boshla", "boshladi", "boshlaydi", "hisoblanadi", "joylashgan",
    # nouns
    "odam", "inson", "bola", "bolalar", "ota", "ona", "aka", "uka", "opa",
    "singil", "do'st", "doʻst", "oila", "uy", "hovli", "shahar", "qishloq",
    "mamlakat", "davlat", "dunyo", "yer", "suv", "havo", "quyosh", "oy",
    "yil", "kun", "hafta", "soat", "vaqt", "payt", "kecha", "bugun", "ertaga",
    "ertalab", "kechqurun", "maktab", "universitet", "talaba", "o'quvchi",
    "oʻquvchi", "o'qituvchi", "oʻqituvchi", "kitob", "kutubxona", "dars",
    "imtihon", "ilm", "fan", "til", "so'z", "soʻz", "gap", "matn", "savol",
    "javob", "ish", "kasb", "pul", "narx", "bozor", "savdo", "iqtisodiyot",
    "sanoat", "qishloq", "xo'jalik", "xoʻjalik", "dehqon", "bog'", "bogʻ",
    "daraxt", "gul", "meva", "olma", "non", "osh", "choy", "yo'l", "yoʻl",
    "ko'cha", "koʻcha", "mahalla", "tog'", "togʻ", "daryo", "dala", "bahor",
    "yoz", "kuz", "qish", "fasl", "ob-havo", "yomg'ir", "yomgʻir", "qor",
    "shamol", "issiq", "sovuq", "tarix", "madaniyat", "san'at", "sanʼat",
    "musiqa", "shoir", "yozuvchi", "asar", "she'r", "sheʼr", "xalq", "millat",
    "vatan", "poytaxt", "aholi", "hukumat", "qonun", "huquq", "sog'liq",
    "sogʻliq", "kasal", "shifokor", "dori", "tibbiyot", "kompyuter", "internet",
    "telefon", "dastur", "ma'lumot", "maʼlumot", "xotira", "tizim", "texnika",
    # adjectives
    "yaxshi", "yomon", "katta", "kichik", "yangi", "eski", "uzun", "qisqa",
    "baland", "past", "keng", "tor", "chiroyli", "go'zal", "goʻzal", "qiyin",
    "oson", "muhim", "zarur", "asosiy", "umumiy", "milliy", "xalqaro", "rasmiy",
    "boy", "kambag'al", "kambagʻal", "tez", "sekin", "to'g'ri", "toʻgʻri",
    "noto'g'ri", "notoʻgʻri", "aniq", "qiziq", "mashhur", "qadimiy",
    # numbers
    "bir", "ikki", "uch", "to'rt", "toʻrt", "besh", "olti", "yetti", "sakkiz",
    "to'qqiz", "toʻqqiz", "o'n", "oʻn", "yigirma", "o'ttiz", "oʻttiz", "qirq",
    "ellik", "oltmish", "yetmish", "sakson", "to'qson", "toʻqson", "yuz",
    "ming", "million", "milliard", "birinchi", "ikkinchi", "uchinchi",
}

# Agglutinative suffixes, longest-first so stripping is greedy.
UZBEK_SUFFIXES = [
    "larimizning", "laringizning", "larimizdan", "laringizdan", "larimizga",
    "laringizga", "larimizda", "laringizda", "yaptilar", "moqdalar", "ganlar",
    "larning", "larimiz", "laringiz", "lardan", "larida", "lariga", "larini",
    "yapman", "yapsan", "yapmiz", "yapsiz", "moqda", "ganman", "gansan",
    "ganmiz", "gansiz", "ning", "ning", "imiz", "ingiz", "lari", "lar",
    "dagi", "digan", "gan", "kan", "qan", "yap", "moq", "ish", "chi",
    "dan", "tan", "ga", "ka", "qa", "da", "ta", "ni", "im", "ing", "si",
    "miz", "ngiz", "di", "ti", "sa", "ib", "ip", "man", "san", "siz", "mas",
]

SUFFIX_DETECT_RE = re.compile(
    r"\w+(lar|ning|ni|ga|da|dan|imiz|ingiz|lari|moqda|yapti|yapman|gan|digan|dagi)\b",
    re.IGNORECASE | re.UNICODE,
)

WORD_RE = re.compile(r"[\wʻʼ']+", re.UNICODE)


def _normalize_apostrophes(text: str) -> str:
    """Fold every apostrophe variant to U+02BB so lexicon lookups match."""
    out = text
    for ch in ASCII_APOSTROPHES:
        out = out.replace(ch, MODIFIER_TURNED_COMMA)
    out = out.replace(MODIFIER_APOSTROPHE, MODIFIER_TURNED_COMMA)
    return out


def _lexicon_normalized() -> set:
    return {_normalize_apostrophes(w.lower()) for w in UZBEK_CORE_WORDS}


_LEXICON = _lexicon_normalized()


def strip_suffixes(word: str) -> list:
    """Return candidate stems, shortest strip first, including the word itself."""
    cands = [word]
    for suf in UZBEK_SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            cands.append(word[: -len(suf)])
    return cands


# ---------------------------------------------------------------------------
# Character trigram plausibility
# ---------------------------------------------------------------------------
class TrigramModel:
    """Character-trigram profile of real Uzbek, for non-word detection."""

    def __init__(self, corpus: list):
        self.trigrams = Counter()
        for para in corpus:
            self._ingest(para)

    def _ingest(self, text: str) -> None:
        norm = _normalize_apostrophes(text.lower())
        for word in WORD_RE.findall(norm):
            padded = "^" + word + "$"
            for i in range(len(padded) - 2):
                self.trigrams[padded[i : i + 3]] += 1

    def unseen_rate(self, text: str) -> float:
        """Fraction of the text's character trigrams absent from real Uzbek.

        High values mean the model is emitting orthographically implausible
        strings -- Uzbek-shaped noise rather than Uzbek words.
        """
        norm = _normalize_apostrophes(text.lower())
        total = 0
        unseen = 0
        for word in WORD_RE.findall(norm):
            padded = "^" + word + "$"
            for i in range(len(padded) - 2):
                total += 1
                if padded[i : i + 3] not in self.trigrams:
                    unseen += 1
        return round(unseen / total, 4) if total else 0.0


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def script_profile(text: str) -> dict:
    lat = len(LATIN_RE.findall(text))
    cyr = len(CYRILLIC_RE.findall(text))
    total = lat + cyr
    if total == 0:
        return {"script": "none", "latin_ratio": 0.0, "cyrillic_ratio": 0.0}
    lr = lat / total
    if lr > 0.9:
        script = "latin"
    elif lr < 0.1:
        script = "cyrillic"
    else:
        script = "mixed"
    return {
        "script": script,
        "latin_ratio": round(lr, 3),
        "cyrillic_ratio": round(1 - lr, 3),
    }


def turkish_contamination(text: str) -> dict:
    hits = [ch for ch in text if ch in TURKISH_ONLY_CHARS]
    letters = max(len(LATIN_RE.findall(text)), 1)
    return {
        "turkish_char_count": len(hits),
        "turkish_char_rate": round(len(hits) / letters, 5),
        "turkish_chars_found": sorted(set(hits)),
    }


def russian_leakage(text: str) -> dict:
    chars = [ch for ch in text if ch in RUSSIAN_ONLY_CHARS]
    words = [w for w in WORD_RE.findall(text.lower()) if w in RUSSIAN_STOPWORDS]
    return {
        "russian_char_count": len(chars),
        "russian_stopword_count": len(words),
    }


def english_leakage(text: str) -> dict:
    words = WORD_RE.findall(text.lower())
    if not words:
        return {"english_stopword_rate": 0.0, "english_stopword_count": 0}
    hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    return {
        "english_stopword_rate": round(hits / len(words), 4),
        "english_stopword_count": hits,
    }


def apostrophe_style(text: str) -> dict:
    """Which oʻ/gʻ convention did the model use, and did it use one at all?"""
    correct = text.count(MODIFIER_TURNED_COMMA) + text.count(MODIFIER_APOSTROPHE)
    ascii_ap = sum(text.count(ch) for ch in ASCII_APOSTROPHES)
    # Bare "o"/"g" where oʻ/gʻ was required is invisible here; we only report
    # which convention appears, plus whether the text has any at all.
    return {
        "modifier_letter_count": correct,
        "ascii_apostrophe_count": ascii_ap,
        "apostrophe_convention": (
            "modifier_letter" if correct > ascii_ap
            else "ascii" if ascii_ap > 0
            else "none"
        ),
    }


def known_word_rate(text: str) -> dict:
    words = [_normalize_apostrophes(w) for w in WORD_RE.findall(text.lower())]
    words = [w for w in words if len(w) > 1 and not w.isdigit()]
    if not words:
        return {"known_word_rate": 0.0, "word_count": 0}
    known = 0
    for w in words:
        if any(stem in _LEXICON for stem in strip_suffixes(w)):
            known += 1
    return {
        "known_word_rate": round(known / len(words), 4),
        "word_count": len(words),
    }


def suffix_rate(text: str) -> dict:
    words = WORD_RE.findall(text)
    if not words:
        return {"suffix_rate": 0.0}
    hits = len(SUFFIX_DETECT_RE.findall(text))
    return {"suffix_rate": round(hits / len(words), 4)}


def repetition_rate(text: str, n: int = 4) -> dict:
    words = WORD_RE.findall(text.lower())
    if len(words) < n * 2:
        return {"repetition_rate": 0.0}
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return {"repetition_rate": round(repeated / len(grams), 4)}


def contains_required(text: str, must_contain_any) -> dict:
    """must_contain_any is a list of groups; each group is satisfied by any member."""
    if not must_contain_any:
        return {"required_coverage": None, "missing_groups": []}
    hay = _normalize_apostrophes(text.lower())
    missing = []
    hit = 0
    for group in must_contain_any:
        variants = [_normalize_apostrophes(str(v).lower()) for v in group]
        if any(v in hay for v in variants):
            hit += 1
        else:
            missing.append(group[0])
    return {
        "required_coverage": round(hit / len(must_contain_any), 3),
        "missing_groups": missing,
    }


def chrf_score(hypothesis: str, reference: str):
    if not reference:
        return None
    try:
        import sacrebleu
    except ImportError:
        return None
    try:
        return round(sacrebleu.sentence_chrf(hypothesis, [reference]).score, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def score_item(item: dict, output: str, trigram_model: TrigramModel) -> dict:
    text = unicodedata.normalize("NFC", output or "")
    scores = {}
    scores.update(script_profile(text))
    scores.update(turkish_contamination(text))
    scores.update(russian_leakage(text))
    scores.update(english_leakage(text))
    scores.update(apostrophe_style(text))
    scores.update(known_word_rate(text))
    scores.update(suffix_rate(text))
    scores.update(repetition_rate(text))
    scores.update(contains_required(text, item.get("must_contain_any")))
    scores["unseen_trigram_rate"] = trigram_model.unseen_rate(text)
    scores["chrf"] = chrf_score(text, item.get("reference") or "")

    # Did it answer in the script we asked for?
    expected_script = item.get("expect_script")
    scores["script_ok"] = (
        None if not expected_script else scores["script"] == expected_script
    )

    # Length floor
    min_words = item.get("min_words")
    scores["length_ok"] = (
        None if not min_words else scores.get("word_count", 0) >= min_words
    )

    # JSON validity, where demanded
    if item.get("expect_json"):
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
        try:
            json.loads(stripped)
            scores["json_valid"] = True
        except (json.JSONDecodeError, ValueError):
            scores["json_valid"] = False

    # Language-appropriate composite flag. For Uzbek answers we want:
    # right script, no Turkish drift, low English leakage, real suffixation.
    if item.get("expect_lang") == "uz":
        scores["uzbek_ok"] = bool(
            scores["script_ok"] is not False
            and scores["turkish_char_count"] == 0
            and scores["english_stopword_rate"] < 0.10
            and scores["repetition_rate"] < 0.25
        )
    else:
        scores["uzbek_ok"] = None

    return scores


def load_suite(path=None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parent / "uzbek_suite.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_trigram_model(suite: dict) -> TrigramModel:
    corpus = list(suite.get("uz_reference_corpus", []))
    # Fold in the Uzbek references and prompts so the profile is a bit broader.
    for item in suite.get("items", []):
        ref = item.get("reference")
        if ref and item.get("expect_lang") == "uz":
            corpus.append(ref)
    return TrigramModel(corpus)
