"""
docx_tracks.py — reads the audio-script tracks of ONE Word document and pairs
each track with its audio file. Standard library only (no python-docx), so it
works — and can be tested — without torch or any model installed.

The table shape is exactly what CustomLayoutGenerator (ProcessAudioScriptWordFile)
and AudioMarkerReview (DocxScriptService) read:

    every table (nested ones included) with >= 2 rows and a non-empty row[0]
      row[0]  = AudioTrackName, e.g. "Mindset_L1_26.mp3"
      row[1]  = the script text

The script text is flattened EXACTLY like AudioMarkerReview's FlattenRow1Text:
for every cell of row[1], every direct paragraph, every direct run: <w:t> text,
<w:br>/<w:tab> -> " ", a "\\n" after each paragraph, and one trailing "\\n"
trimmed. That matters: the review tool refuses a draft whose word count differs
from the word count it reads from the same table.
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")

# "[1234]", "[1:02.5]", "[01:02:03]" — the timing markers CustomLayoutGenerator reads.
MARKER_RE = re.compile(r"\[(?:\d{1,2}:)?(?:\d{1,2}:)?\d+(?:\.\d+)?\]")


@dataclass
class Track:
    index: int                 # 0-based position among qualifying tables (document order)
    name: str                  # row[0] caption, e.g. "Mindset_L1_26.mp3"
    text: str                  # row[1] flattened text (see module docstring)
    number: int | None = None  # the number the user types, e.g. 26
    audio: Path | None = None  # matched audio file, if any
    note: str = ""             # why no audio was matched (ambiguous / not found)
    has_markers: bool = False  # row[1] already contains [ms] markers
    not_spoken: set[int] = field(default_factory=set)  # word indices styled "NotSpoken"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def stem(self) -> str:
        """Output file stem: the audio file's stem (what the review tool expects),
        or the caption without its extension when no audio was matched."""
        if self.audio is not None:
            return self.audio.stem
        return strip_audio_ext(self.name).strip() or f"track_{self.index + 1:02d}"


# ---------------------------------------------------------------------------
# Reading the document
# ---------------------------------------------------------------------------

def _row_caption(row) -> str:
    return "".join(t.text or "" for t in row.iter(W + "t")).strip()


def _not_spoken_style_ids(z: zipfile.ZipFile) -> set[str]:
    """Ids of the character styles named "NotSpoken" (any spacing/case), like
    CustomLayoutGenerator's IsTargetStyle."""
    try:
        styles = ET.fromstring(z.read("word/styles.xml"))
    except KeyError:
        return set()
    ids = set()
    for st in styles.iter(W + "style"):
        sid = st.get(W + "styleId") or ""
        name_el = st.find(W + "name")
        name = name_el.get(W + "val") if name_el is not None else ""
        if any(v.replace(" ", "").lower() == "notspoken" for v in (sid, name or "")):
            ids.add(sid)
    return ids


def _flatten_row1(row, not_spoken_ids: set[str] | None = None) -> tuple[str, set[int]]:
    """(text, indices of the words written entirely in a NotSpoken-styled run)."""
    parts: list[str] = []
    flags: list[bool] = []  # per character: comes from a NotSpoken run
    not_spoken_ids = not_spoken_ids or set()

    def add(s: str, ns: bool) -> None:
        parts.append(s)
        flags.extend([ns] * len(s))

    for cell in row.findall(W + "tc"):
        for para in cell.findall(W + "p"):
            for run in para.findall(W + "r"):
                rstyle = run.find(W + "rPr/" + W + "rStyle")
                ns = rstyle is not None and rstyle.get(W + "val") in not_spoken_ids
                for child in run:
                    if child.tag == W + "t":
                        add(child.text or "", ns)
                    elif child.tag in (W + "br", W + "tab"):
                        add(" ", False)
            add("\n", False)
    text = "".join(parts)
    if text.endswith("\n"):
        text = text[:-1]
    words = set()
    for i, m in enumerate(re.finditer(r"\S+", text)):
        if all(flags[m.start():m.end()]):
            words.add(i)
    return text, words


def read_tracks(docx_path: Path) -> list[Track]:
    with zipfile.ZipFile(docx_path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        ns_ids = _not_spoken_style_ids(z)
    tracks: list[Track] = []
    for table in root.iter(W + "tbl"):
        rows = table.findall(W + "tr")
        if len(rows) < 2:
            continue
        name = _row_caption(rows[0])
        if not name:
            continue
        text, not_spoken = _flatten_row1(rows[1], ns_ids)
        tracks.append(Track(index=len(tracks), name=name, text=text,
                            has_markers=bool(MARKER_RE.search(text)), not_spoken=not_spoken))
    _assign_numbers(tracks)
    return tracks


def _assign_numbers(tracks: list[Track]) -> None:
    """The number the user types: the LAST group of digits in the caption
    ("Mindset_L1_26.mp3" -> 26). If captions have no numbers, or two captions
    share one, every track falls back to its position (1, 2, 3, ...)."""
    numbers = []
    for t in tracks:
        digits = re.findall(r"\d+", strip_audio_ext(t.name))
        numbers.append(int(digits[-1]) if digits else None)
    if all(n is not None for n in numbers) and len(set(numbers)) == len(numbers):
        for t, n in zip(tracks, numbers):
            t.number = n
    else:
        for t in tracks:
            t.number = t.index + 1


# ---------------------------------------------------------------------------
# Matching tracks to audio files
# ---------------------------------------------------------------------------

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def strip_audio_ext(name: str) -> str:
    low = name.lower()
    for ext in AUDIO_EXTS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def normalize_name(name: str) -> str:
    """Same idea as AudioMarkerReview: lowercase, drop an "audio:" prefix and the
    extension, Persian/Arabic digits -> ASCII, keep only letters and digits."""
    n = name.strip().translate(_PERSIAN_DIGITS)
    if n.lower().startswith("audio:"):
        n = n[6:]
    n = strip_audio_ext(n.strip())
    return re.sub(r"[^0-9a-z]", "", n.lower())


def _digits(n: str) -> list[str]:
    return [d.lstrip("0") or "0" for d in re.findall(r"\d+", n)]


def match_audio(tracks: list[Track], audio_dir: Path) -> None:
    files = sorted(p for p in audio_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS) if audio_dir.is_dir() else []
    by_norm: dict[str, list[Path]] = {}
    for p in files:
        by_norm.setdefault(normalize_name(p.name), []).append(p)

    for t in tracks:
        t.audio, t.note = None, ""
        key = normalize_name(t.name)
        exact = by_norm.get(key, [])
        if len(exact) == 1:
            t.audio = exact[0]
            continue
        if len(exact) > 1:
            t.note = "more than one audio file: " + ", ".join(p.name for p in exact)
            continue
        # Containment either way, but numbers must agree exactly, so
        # "unit3" never matches "unit13" and "L1_02" never matches "L1_20".
        cands = [p for norm, ps in by_norm.items() for p in ps
                 if norm and key and (norm in key or key in norm) and _digits(norm) == _digits(key)]
        if len(cands) == 1:
            t.audio = cands[0]
        elif len(cands) > 1:
            t.note = "ambiguous: " + ", ".join(p.name for p in cands)
        else:
            t.note = "no audio file found"


# ---------------------------------------------------------------------------
# "26", "26-31", "4, 7-9 12", "all", "new"
# ---------------------------------------------------------------------------

def parse_selection(spec: str, tracks: list[Track], done: set[int] | None = None) -> list[Track]:
    """Returns the selected tracks in document order. Raises ValueError with a
    readable message for anything it can't understand."""
    spec = spec.strip().lower().translate(_PERSIAN_DIGITS)
    by_number = {t.number: t for t in tracks}
    if spec in ("all", "*"):
        return list(tracks)
    if spec == "new":
        done = done or set()
        return [t for t in tracks if t.index not in done]

    wanted: set[int] = set()
    for part in re.split(r"[,\s،]+", spec):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            wanted.update(range(a, b + 1))
        elif part.isdigit():
            wanted.add(int(part))
        else:
            raise ValueError(f"don't understand '{part}' (examples: 26   26-31   4,7-9   all   new)")

    missing = sorted(n for n in wanted if n not in by_number)
    chosen = [by_number[n] for n in sorted(wanted) if n in by_number]
    if not chosen:
        raise ValueError("none of these numbers exist in the document"
                         + (f": {', '.join(map(str, missing))}" if missing else ""))
    if missing:
        # Ranges may cross numbers that simply have no table (e.g. no script for 30).
        print(f"  (no table for: {', '.join(map(str, missing))} — skipped)")
    return sorted(chosen, key=lambda t: t.index)
