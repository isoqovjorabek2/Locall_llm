"""Version-comparison checks for require_gemma4_support().

The guard must never call a NEWER transformers "too old". Naive string
comparison does exactly that: "5.15" sorts below "5.5".

    python tests/test_version_guard.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.common import MIN_TRANSFORMERS  # noqa: E402

# Reuse the exact parser from the guard rather than a copy, so this test cannot
# drift away from the code it is checking.
_SRC = (pathlib.Path(__file__).resolve().parent.parent / "bench" / "common.py").read_text(
    encoding="utf-8"
)
_BODY = _SRC[_SRC.index("    def _parse(v):") : _SRC.index("    installed = _parse(version)")]
_NS: dict = {}
exec("\n".join(line[4:] for line in _BODY.splitlines()), _NS)
parse = _NS["_parse"]

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(name + "  " + detail)
    print("  [" + status + "] " + name + ("  " + detail if detail else ""))


def main():
    req = parse(MIN_TRANSFORMERS)

    print("\n--- the regression this exists for ---")
    check("string compare really is wrong", "5.15" < "5.5",
          "'5.15' < '5.5' is " + str("5.15" < "5.5"))
    check("5.15.0 is accepted", parse("5.15.0") >= req, str(parse("5.15.0")))
    check("5.15 (two-part) is accepted", parse("5.15") >= req, str(parse("5.15")))
    check("5.15 outranks 5.5", parse("5.15") > parse("5.5"))

    print("\n--- accepted versions ---")
    for v in ["5.5.0", "5.6.0", "5.15.0", "5.99.0", "6.0.0", "10.0.0"]:
        check(v + " accepted", parse(v) >= req, str(parse(v)))

    print("\n--- rejected versions ---")
    for v in ["4.57.6", "5.4.9", "4.99.0", "0.1.0"]:
        check(v + " rejected", parse(v) < req, str(parse(v)))

    print("\n--- suffixes must not break parsing ---")
    check("5.5.0rc1 parses to (5,5,0)", parse("5.5.0rc1") == (5, 5, 0), str(parse("5.5.0rc1")))
    check("5.15.1.dev0 accepted", parse("5.15.1.dev0") >= req, str(parse("5.15.1.dev0")))
    check("5.5.0+cu128 accepted", parse("5.5.0+cu128") >= req, str(parse("5.5.0+cu128")))

    print("\n--- ordering is monotonic ---")
    ordered = ["4.57.0", "5.0.0", "5.4.9", "5.5.0", "5.6.0", "5.15.0", "6.0.0"]
    pairs = list(zip(ordered, ordered[1:]))
    check("each version outranks the previous",
          all(parse(a) < parse(b) for a, b in pairs))

    print("\n" + "=" * 60)
    if FAILURES:
        print(str(len(FAILURES)) + " CHECK(S) FAILED:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("All checks passed. MIN_TRANSFORMERS = " + MIN_TRANSFORMERS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
