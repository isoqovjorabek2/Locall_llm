"""Sanity checks for the Uzbek scorers.

The point is discrimination: each metric must separate genuine Uzbek from the
specific failure mode it exists to catch. Run with:

    python tests/test_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.uz_scoring import (  # noqa: E402
    build_trigram_model,
    load_suite,
    score_item,
)

SUITE = load_suite()
TRI = build_trigram_model(SUITE)

# --- samples -------------------------------------------------------------
GOOD_UZ = (
    "Toshkent Oʻzbekistonning poytaxti boʻlib, mamlakatning eng yirik shahri "
    "hisoblanadi. Shaharda koʻplab universitetlar, kutubxonalar va muzeylar "
    "mavjud. Aholisi ikki yarim milliondan ortiq odamni tashkil qiladi."
)
GOOD_UZ_ASCII_APOS = (
    "Toshkent O'zbekistonning poytaxti bo'lib, mamlakatning eng yirik shahri "
    "hisoblanadi. Shaharda ko'plab universitetlar va kutubxonalar mavjud."
)
TURKISH_DRIFT = (
    "Taşkent Özbekistan'ın başkentidir ve ülkenin en büyük şehridir. "
    "Şehirde birçok üniversite ve kütüphane bulunmaktadır."
)
ENGLISH_ANSWER = (
    "Tashkent is the capital of Uzbekistan and it is the largest city in the "
    "country. There are many universities and libraries in the city."
)
CYRILLIC_UZ = (
    "Тошкент Ўзбекистоннинг пойтахти бўлиб, мамлакатнинг энг йирик шаҳри "
    "ҳисобланади. Шаҳарда кўплаб университетлар ва кутубхоналар мавжуд."
)
NONWORD_SALAD = (
    "Toshqvent Ozbekxstonnzng poytqxti bolib, mamlqkatnzng enq yirzk shqhri "
    "hzsoblqnadi. Shqhardq koplqb unzversztetlqr vq kutubxonqlqr mqvjud."
)
LOOPING = ("Toshkent Oʻzbekiston poytaxti. " * 12)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(name + "  " + detail)
    print("  [" + status + "] " + name + ("  " + detail if detail else ""))


def s(text, expect_script="latin", expect_lang="uz", **extra):
    item = {"expect_script": expect_script, "expect_lang": expect_lang}
    item.update(extra)
    return score_item(item, text, TRI)


def main():
    print("\n--- script detection ---")
    check("good Uzbek Latin -> latin", s(GOOD_UZ)["script"] == "latin")
    check("Cyrillic Uzbek -> cyrillic",
          s(CYRILLIC_UZ, expect_script="cyrillic")["script"] == "cyrillic")
    check("Cyrillic answer flagged when Latin requested",
          s(CYRILLIC_UZ, expect_script="latin")["script_ok"] is False)

    print("\n--- Turkish contamination ---")
    good_tr = s(GOOD_UZ)["turkish_char_count"]
    bad_tr = s(TURKISH_DRIFT)["turkish_char_count"]
    check("clean Uzbek has zero Turkish chars", good_tr == 0, "got " + str(good_tr))
    check("Turkish text is caught", bad_tr > 0, "got " + str(bad_tr))
    check("Turkish drift fails uzbek_ok", s(TURKISH_DRIFT)["uzbek_ok"] is False)

    print("\n--- English leakage ---")
    good_en = s(GOOD_UZ)["english_stopword_rate"]
    bad_en = s(ENGLISH_ANSWER)["english_stopword_rate"]
    check("Uzbek has low English rate", good_en < 0.05, "got " + str(good_en))
    check("English answer has high rate", bad_en > 0.20, "got " + str(bad_en))
    check("English answer fails uzbek_ok", s(ENGLISH_ANSWER)["uzbek_ok"] is False)

    print("\n--- non-word detection (the Bonsai failure mode) ---")
    good_tri = s(GOOD_UZ)["unseen_trigram_rate"]
    bad_tri = s(NONWORD_SALAD)["unseen_trigram_rate"]
    check("real Uzbek has low unseen-trigram rate", good_tri < 0.20,
          "got " + str(good_tri))
    check("non-word salad scores much worse", bad_tri > good_tri * 2,
          "good=" + str(good_tri) + " bad=" + str(bad_tri))

    good_kw = s(GOOD_UZ)["known_word_rate"]
    bad_kw = s(NONWORD_SALAD)["known_word_rate"]
    check("real Uzbek has higher known-word rate", good_kw > bad_kw,
          "good=" + str(good_kw) + " bad=" + str(bad_kw))

    print("\n--- repetition ---")
    good_rep = s(GOOD_UZ)["repetition_rate"]
    bad_rep = s(LOOPING)["repetition_rate"]
    check("normal text is not flagged as looping", good_rep < 0.10,
          "got " + str(good_rep))
    check("looping text is flagged", bad_rep > 0.5, "got " + str(bad_rep))

    print("\n--- apostrophe convention ---")
    check("modifier letters detected",
          s(GOOD_UZ)["apostrophe_convention"] == "modifier_letter")
    check("ASCII apostrophes detected",
          s(GOOD_UZ_ASCII_APOS)["apostrophe_convention"] == "ascii")
    check("both spellings score alike on known words",
          abs(s(GOOD_UZ)["known_word_rate"] - s(GOOD_UZ_ASCII_APOS)["known_word_rate"]) < 0.20,
          "modifier=" + str(s(GOOD_UZ)["known_word_rate"])
          + " ascii=" + str(s(GOOD_UZ_ASCII_APOS)["known_word_rate"]))

    print("\n--- required-content matching ---")
    hit = s(GOOD_UZ, must_contain_any=[["Toshkent", "Тошкент"]])["required_coverage"]
    miss = s(GOOD_UZ, must_contain_any=[["Samarqand"]])["required_coverage"]
    check("present term matches", hit == 1.0, "got " + str(hit))
    check("absent term does not match", miss == 0.0, "got " + str(miss))
    check("apostrophe-insensitive matching",
          s(GOOD_UZ_ASCII_APOS,
            must_contain_any=[["Oʻzbekiston"]])["required_coverage"] == 1.0)

    print("\n--- composite ---")
    check("clean Uzbek passes uzbek_ok", s(GOOD_UZ)["uzbek_ok"] is True)
    check("ASCII-apostrophe Uzbek still passes", s(GOOD_UZ_ASCII_APOS)["uzbek_ok"] is True)

    print("\n--- suite integrity ---")
    ids = [it["id"] for it in SUITE["items"]]
    check("no duplicate item ids", len(ids) == len(set(ids)))
    check("every item has a category",
          all(it.get("category") for it in SUITE["items"]))
    check("every item has a prompt", all(it.get("prompt") for it in SUITE["items"]))
    check("corpus is non-empty", len(SUITE["uz_reference_corpus"]) >= 5)
    print("  (suite has " + str(len(ids)) + " items across "
          + str(len({it["category"] for it in SUITE["items"]})) + " categories)")

    print("\n" + "=" * 60)
    if FAILURES:
        print(str(len(FAILURES)) + " CHECK(S) FAILED:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
