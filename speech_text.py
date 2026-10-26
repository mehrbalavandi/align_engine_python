"""
speech_text.py — which words of a script are actually spoken, and how to spell the
spoken ones for the aligner. Standard library only (testable without torch).

1. Unspoken words (not aligned, never highlighted in the app):
   - a speaker label at the start of a paragraph: 1-3 words, the first starting with a
     capital letter, the last ending with ":" ("Man:", "Bank clerk:", "Candidate 1:"),
     at most 24 characters, none of them a common function word ("The answer is:" is
     spoken, not a label). Same rule as CustomLayoutGenerator's AudioScriptTiming (A-2).
   - any text with the Word character style "NotSpoken" (read in docx_tracks.py).
   - the automatic number of a bulleted Word list paragraph, or of a numbered one whose
     number is not numeric ("A:", "iv."): Word draws it, so it is not in the runs' text.
     docx_tracks.py puts LIST_NUMBER_TOKEN ("#") in its place, styled NotSpoken (A-4).
     A NUMERIC list number ("1:", "2.", "3") is spoken instead: docx_tracks.py writes the
     number itself and the aligner hears "One." (A-4b, list_number_token).
     AudioMarkerReview and CustomLayoutGenerator use the same token/rule.
   They get the silence (or unscripted speech) between the words around them, so the
   neighbours keep their real start/end instead of the label stealing time from them.

2. Numbers are spelled out for the aligner: the model only knows letters, so "1",
   "$10.50" or "1985." used to get zero length and were never highlighted.
   "1" -> "one", "$10.50" -> "tendollarsfifty", "1985." -> "nineteeneightyfive",
   "21st" -> "twentyfirst", "25,000" -> "twentyfivethousand". The words of one token
   are written together so the aligner still sees exactly one word per token.
"""

from __future__ import annotations

import re

# A-4: stands for a Word list paragraph's automatic number when it is not numeric (see
# docx_tracks._flatten_row1); a numeric one is written as the number itself (A-4b).
LIST_NUMBER_TOKEN = "#"
# A list-number word at the start of a line, before a speaker label: "# Man: ...", "1: Man: ..."
_LEADING_LIST_TOKEN = re.compile(r"^\s*(?:" + re.escape(LIST_NUMBER_TOKEN) + r"|\d+[.:)]?)(?=\s)")


def list_number_token(marker: str | None) -> str:
    """A-4b: the word that stands for a Word list paragraph's automatic number.
    Numeric ("1:", "2.", "3", "1.2") -> the number without spaces, spoken like any word;
    anything else (bullet, "A:", "iv.", unknown) -> LIST_NUMBER_TOKEN, unspoken."""
    if marker and any(ch.isdigit() for ch in marker):
        token = re.sub(r"\s+", "", marker)
        if token:
            return token
    return LIST_NUMBER_TOKEN

LABEL_RE = re.compile(r"^\s*[A-Z][\w'’.\-]*(?:[ \t]+[\w'’.\-]+){0,2}:(?=\s|$)")
MAX_LABEL_LENGTH = 24
# A spoken phrase ending in ":" ("The answer is:") is not a label: labels are names and roles.
LABEL_STOPWORDS = frozenset(
    "is are was were be been am the a an my your his her its our their this that these those "
    "it i you we they he she and or but to of in on for with at as here there what which who "
    "so if then do does did have has had not no yes".split())


def is_label(prefix: str) -> bool:
    words = prefix.strip().rstrip(":").split()
    if len(prefix.strip()) > MAX_LABEL_LENGTH:
        return False
    if len(words) == 1:
        return True  # "Man:", "A:", "Examiner:"
    # "Student A:" is a label; a single capital letter is a name, not the article "a".
    return not any(w.lower().strip("'’.-") in LABEL_STOPWORDS
                   and not (len(w) == 1 and w.isupper()) for w in words)


def label_word_indices(text: str) -> set[int]:
    """Word indices (text.split() order) of the speaker labels at paragraph starts."""
    out: set[int] = set()
    index = 0
    for line in text.split("\n"):
        words = line.split()
        # A-4: "# Man: ..." — the list-number token comes before the label
        lead = _LEADING_LIST_TOKEN.match(line)
        body = line[lead.end():] if lead else line
        first = index + (1 if lead else 0)
        m = LABEL_RE.match(body)
        if m and is_label(m.group(0)):
            n = len(m.group(0).split())
            out.update(range(first, first + n))
        index += len(words)
    return out


# ---------------------------------------------------------------------------
# Numbers -> words
# ---------------------------------------------------------------------------

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                      "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
_CURRENCY = {"$": ("dollars", "cents"), "£": ("pounds", "pence"), "€": ("euros", "cents")}


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + cardinal(rest))
    for value, name in _SCALES:
        if n >= value:
            rest = n % value
            head = cardinal(n // value) + " " + name
            if rest == 0:
                return head
            return head + (" and " if rest < 100 else " ") + cardinal(rest)
    return str(n)


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(n: int) -> str:
    if 2000 <= n <= 2009:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + (" oh " + cardinal(lo) if lo < 10 else " " + cardinal(lo))


def _number_words(core: str) -> str | None:
    """Words for a bare numeric string (no currency sign), or None if it isn't one."""
    m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+)(st|nd|rd|th)?", core)
    if m:
        n = int(m.group(1).replace(",", ""))
        if m.group(2):
            return ordinal(n)
        if "," not in m.group(1) and len(m.group(1)) == 4 and 1100 <= n <= 2099:
            return year(n)
        return cardinal(n)
    m = re.fullmatch(r"(\d+)s", core)  # 1980s
    if m:
        n = int(m.group(1))
        w = year(n) if len(m.group(1)) == 4 else cardinal(n)
        return w + "s"
    m = re.fullmatch(r"(\d+)[.:](\d{2})", core)  # 7.50 pm, 10:30
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return cardinal(a) + ("" if b == 0 else " " + (("oh " + cardinal(b)) if b < 10 else cardinal(b)))
    m = re.fullmatch(r"(\d+)\.(\d+)", core)  # 2.5
    if m:
        return cardinal(int(m.group(1))) + " point " + " ".join(_ONES[int(d)] for d in m.group(2))
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", core)  # 1-6
    if m:
        return cardinal(int(m.group(1))) + " to " + cardinal(int(m.group(2)))
    m = re.fullmatch(r"(\d+)%", core)
    if m:
        return cardinal(int(m.group(1))) + " percent"
    return None


def speakable(token: str) -> str:
    """The token as the aligner should see it. Tokens without digits are returned as they
    are; tokens with digits are spelled out (letters only, no spaces) when possible."""
    if not any(ch.isdigit() for ch in token):
        return token
    core = token.strip("\"'“”‘’()[]{}!?;,.…")
    core = core.rstrip(":")
    sign = core[:1] if core[:1] in _CURRENCY else ""
    if sign:
        core = core[1:]
    words = None
    if sign:
        m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?", core)
        if m:
            main, minor = _CURRENCY[sign]
            words = cardinal(int(m.group(1).replace(",", ""))) + " " + main
            if m.group(2) and int(m.group(2)):
                words += " " + cardinal(int(m.group(2)))
    else:
        words = _number_words(core)
    if words is None:
        # Letters and digits mixed in an unknown way: spell every digit group.
        words = re.sub(r"\d+", lambda d: " " + cardinal(int(d.group(0))) + " ", core)
    return re.sub(r"[^a-z]", "", words.lower()) or token
