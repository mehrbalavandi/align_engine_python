"""
align_batch.py — word-level forced alignment for the IELTS audio scripts.

Step 1 of the audio pipeline:
    align_batch.py  ->  <stem>_draft.json  ->  AudioMarkerReview  ->  Word with [ms] markers
                                                                  ->  CustomLayoutGenerator

Just run it (F5 in VS Code, run.bat, or `python align_batch.py`). It asks for:
  1. the Word document with all the audio-script tables (e.g. "Mindset1- Audio Scripts.docx")
  2. the folder with that book's audio files
and then, again and again, which track numbers to align ("26", "26-31", "4,7-9",
"all", "new"). Both paths are remembered in align_config.json for next time.
To change the saved values: S in the menu, settings.bat, `--settings`, or `--reset`
(forget everything and ask again); see GUIDE.md, "Changing the saved settings".

For every chosen track it writes, into the output folder (default: <audio folder>/output):
    <audio stem>_draft.json    <- open in AudioMarkerReview (it finds it by itself)
    <audio stem>_markers.txt   <- the same timings as "[ms]word[ms]..." text, for a quick look

Non-interactive (for scripts):
    python align_batch.py --docx "Mindset1- Audio Scripts.docx" --audio_dir audio --tracks 26-31
    python align_batch.py --docx ... --audio_dir ... --tracks new --dry-run      (only list)

Old mode (txt/docx + mp3 pairs with the same name in one folder):
    python align_batch.py --pairs                 (uses the "share" folder next to this script)
    python align_batch.py --pairs --share_dir D:\\work\\share

See GUIDE.md for the whole workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from docx_tracks import AUDIO_EXTS, Track, match_audio, parse_selection, read_tracks

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "align_config.json"
DEFAULT_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
LOW_CONFIDENCE = -1.5  # words below this score are worth a look in AudioMarkerReview


# ---------------------------------------------------------------------------
# Settings remembered between runs
# ---------------------------------------------------------------------------

def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"  (could not save {CONFIG_PATH.name}: {e})")


# ---------------------------------------------------------------------------
# Alignment (torch is imported only when something is actually aligned, so
# listing tracks and --dry-run start instantly)
# ---------------------------------------------------------------------------

def is_out_of_memory(e: BaseException) -> bool:
    """RAM/VRAM ran out while running the model (CPU: "DefaultCPUAllocator: not enough memory",
    CUDA: "CUDA out of memory")."""
    if isinstance(e, MemoryError):
        return True
    msg = str(e).lower()
    return "not enough memory" in msg or "out of memory" in msg or "defaultcpuallocator" in msg


class Aligner:
    """batch_size = how many 30-second pieces of audio go through the model at once.

    Memory grows with it: on CPU each piece needs roughly 0.5-1 GB at the model's first layer,
    so 8 pieces (only reached by tracks longer than 4 minutes) asked for ~1.8 GB in a single
    block and failed on machines with little free RAM. Batching barely speeds things up on CPU,
    so CPU uses 2 by default. If memory still runs out, align() retries the same track with a
    smaller batch (8 -> 4 -> 2 -> 1) and then shorter pieces (30 s -> 15 s), and keeps the
    smaller setting for the rest of the run. The result is the same either way.
    """

    MIN_WINDOW = 10  # seconds; shorter pieces start to hurt accuracy at their edges

    def __init__(self, device: str | None, model_id: str, language: str,
                 batch_size: int | None = None, window_size: int = 30, context_size: int = 2):
        self.device_arg, self.model_id, self.language = device, model_id, language
        self.batch_size, self.window_size, self.context_size = batch_size, window_size, context_size
        self._model = None
        self._tokenizer = None
        self._lib = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        import ctc_forced_aligner as lib
        device = self.device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        if self.batch_size is None:
            self.batch_size = 8 if device == "cuda" else 2
        print(f"Loading the alignment model on {device} (first time: downloads a few hundred MB) ...")
        self._model, self._tokenizer = lib.load_alignment_model(device, self.model_id, dtype=dtype)
        self._lib = lib

    def _free_memory(self):
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _shrink(self) -> bool:
        """One step smaller after running out of memory; False = nothing left to shrink."""
        if self.batch_size > 1:
            self.batch_size = max(1, self.batch_size // 2)
            print(f"    (not enough memory - retrying with {self.batch_size} piece(s) at a time)")
            return True
        if self.window_size // 2 >= self.MIN_WINDOW:
            self.window_size //= 2
            print(f"    (not enough memory - retrying with {self.window_size}-second pieces)")
            return True
        return False

    def _emissions(self, waveform):
        while True:
            try:
                return self._lib.generate_emissions(
                    self._model, waveform, self.window_size, self.context_size, self.batch_size)
            except (RuntimeError, MemoryError) as e:
                if not is_out_of_memory(e):
                    raise
                self._free_memory()
                if not self._shrink():
                    raise MemoryError(
                        "not enough free memory even with the smallest settings. Close other "
                        "programs (browser, Word, VS Code extensions) and try this track again.") from e

    def align(self, audio_path: Path, original_text: str) -> list:
        """Word results in text order: index i == word i of original_text.split()."""
        self._load()
        lib = self._lib
        flat_text = original_text.replace("\n", " ").strip()
        waveform = lib.load_audio(str(audio_path), self._model.dtype, self._model.device)
        emissions, stride = self._emissions(waveform)
        del waveform
        tokens_starred, text_starred = lib.preprocess_text(
            flat_text, romanize=True, language=self.language, split_size="word", star_frequency="edges")
        segments, scores, blank_token = lib.get_alignments(emissions, tokens_starred, self._tokenizer)
        spans = lib.get_spans(tokens_starred, segments, blank_token)
        results = lib.postprocess_results(text_starred, spans, stride, scores)

        expected = flat_text.split()
        if len(results) != len(expected):
            raise ValueError(
                f"the aligner returned {len(results)} words but the text has {len(expected)}. "
                "Usually the text and the audio don't match (missing/extra words) or the wrong "
                "audio file was paired.")
        return results


# ---------------------------------------------------------------------------
# "[ms]word[ms]word..." text, keeping the original whitespace and line breaks.
# One marker == shared boundary (end of the previous word AND start of the next),
# placed in the middle of the silence between them.
# ---------------------------------------------------------------------------

def build_boundaries(results: list) -> tuple[list, list, list]:
    n = len(results)
    if n == 0:
        raise ValueError("no aligned words")
    b = [0] * (n + 1)
    b[0] = round(results[0]["start"] * 1000)
    for i in range(1, n):
        b[i] = round(((results[i - 1]["end"] + results[i]["start"]) / 2) * 1000)
    b[n] = round(results[-1]["end"] * 1000)
    words = [r["text"] for r in results]
    scores = [round(r["score"], 4) for r in results]
    return words, b, scores


def build_marker_text(original_text: str, boundaries_ms: list, keep: set | None = None) -> str:
    """keep=None -> a marker after every word; otherwise only boundaries in `keep`."""
    out, wi = [], 0
    for tok in re.findall(r"\S+|\s+", original_text):
        if tok.strip() == "":
            out.append(tok)
            continue
        if wi == 0:
            out.append(f"[{boundaries_ms[0]}]")
        out.append(tok)
        if keep is None or (wi + 1) in keep:
            out.append(f"[{boundaries_ms[wi + 1]}]")
        wi += 1
    return "".join(out)


def sentence_boundaries(words: list) -> set:
    end = re.compile(r"[.!?]\"?'?$")
    keep = {0, len(words)}
    keep.update(i + 1 for i, w in enumerate(words) if end.search(w))
    return keep


def write_outputs(out_dir: Path, stem: str, audio_name: str, text: str,
                  results: list, granularity: str) -> int:
    words, boundaries, scores = build_boundaries(results)
    keep = sentence_boundaries(words) if granularity == "sentence" else None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_markers.txt").write_text(
        build_marker_text(text, boundaries, keep), encoding="utf-8")
    draft = {"audioFile": audio_name, "words": words, "boundariesMs": boundaries, "scores": scores}
    (out_dir / f"{stem}_draft.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    return sum(1 for s in scores if s < LOW_CONFIDENCE)


# ---------------------------------------------------------------------------
# Word document + audio folder mode
# ---------------------------------------------------------------------------

def default_out_dir(audio_dir: Path) -> Path:
    return audio_dir / "output"


def done_indices(tracks: list[Track], out_dir: Path) -> set[int]:
    return {t.index for t in tracks if (out_dir / f"{t.stem}_draft.json").exists()}


def print_tracks(tracks: list[Track], done: set[int]) -> None:
    print(f"\n  {'No.':>4}  {'track':<28} {'words':>6}  audio / status")
    for t in tracks:
        if t.has_markers:
            status = "! text already has [ms] markers — use the unmarked document"
        elif t.audio is None:
            status = f"! {t.note}"
        else:
            status = t.audio.name + ("   (aligned)" if t.index in done else "")
        print(f"  {t.number:>4}  {t.name[:28]:<28} {t.word_count:>6}  {status}")
    print()


def summary_line(tracks: list[Track], done: set[int]) -> str:
    if not tracks:
        return "no audio-script tables found"
    nums = [t.number for t in tracks]
    with_audio = sum(1 for t in tracks if t.audio is not None and not t.has_markers)
    return (f"{len(tracks)} tracks ({min(nums)}–{max(nums)}), {with_audio} ready to align, "
            f"{len(done)} already aligned")


def align_tracks(selected: list[Track], aligner: Aligner, out_dir: Path, granularity: str,
                 overwrite: bool | None, dry_run: bool = False) -> tuple[int, list]:
    """overwrite: True / False / None (= ask once if some outputs already exist)."""
    runnable = [t for t in selected if t.audio is not None and not t.has_markers]
    for t in selected:
        if t not in runnable:
            reason = "text already has markers" if t.has_markers else t.note
            print(f"  [skip] {t.number} {t.name}: {reason}")

    existing = [t for t in runnable if (out_dir / f"{t.stem}_draft.json").exists()]
    if existing and overwrite is None:
        nums = ", ".join(str(t.number) for t in existing)
        ans = input(f"  {len(existing)} already aligned ({nums}). Overwrite them? [y/N] ").strip().lower()
        overwrite = ans in ("y", "yes")
    if existing and not overwrite:
        runnable = [t for t in runnable if t not in existing]
        print(f"  keeping the existing outputs of {len(existing)} track(s)")

    if dry_run:
        for t in runnable:
            print(f"  would align {t.number}: {t.audio.name}  ({t.word_count} words)"
                  f"  ->  {out_dir / (t.stem + '_draft.json')}")
        return 0, []
    if not runnable:
        print("  nothing to align")
        return 0, []

    ok, failed = 0, []
    for i, t in enumerate(runnable, 1):
        print(f"[{i}/{len(runnable)}] {t.number}: {t.audio.name} ({t.word_count} words) ...")
        try:
            results = aligner.align(t.audio, t.text)
            low = write_outputs(out_dir, t.stem, t.audio.name, t.text, results, granularity)
            note = f", {low} low-confidence word(s) to check" if low else ""
            print(f"    -> {t.stem}_draft.json{note}")
            ok += 1
        except KeyboardInterrupt:
            print("\n  stopped by you (Ctrl+C)")
            break
        except ModuleNotFoundError as e:
            print(f"    [STOPPED] {e}.\n    The Python packages aren't installed in this environment: run "
                  "setup.ps1 once, and run the script with run.bat (or select the .venv interpreter "
                  "in VS Code). See GUIDE.md, section 1.")
            failed.extend(runnable[i - 1:])
            break
        except Exception as e:  # noqa: BLE001 — keep going with the next track
            print(f"    [FAILED] {e}")
            failed.append(t)
    print(f"\nDone: {ok} aligned, {len(failed)} failed. Output folder: {out_dir}")
    if failed:
        print("Failed: " + ", ".join(f"{t.number} ({t.name})" for t in failed))
    return ok, failed


def ask_path(prompt: str, current: str | None, want_dir: bool) -> Path | None:
    """Enter = keep current; '?' (or Enter when nothing is set) = Windows picker;
    a typed/dragged path is also fine (quotes and PowerShell's '& ' are removed)."""
    hint = f" [Enter = {current}]" if current else " [Enter = open a picker]"
    raw = input(f"{prompt}{hint}: ").strip()
    if raw == "" and current:
        return Path(current)
    if raw in ("", "?"):
        picked = pick_with_dialog(prompt, want_dir)
        return Path(picked) if picked else (Path(current) if current else None)
    raw = re.sub(r"^&\s*", "", raw).strip().strip('"').strip("'")
    return Path(raw).expanduser()


def pick_with_dialog(title: str, want_dir: bool) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if want_dir:
            path = filedialog.askdirectory(title=title)
        else:
            path = filedialog.askopenfilename(title=title, filetypes=[("Word document", "*.docx")])
        root.destroy()
        return path or None
    except Exception as e:  # noqa: BLE001 — no GUI available: fall back to typing
        print(f"  (no file picker available: {e}; type or paste the path instead)")
        return None


def setup_paths(cfg: dict, force: bool = False) -> bool:
    if force:
        # show what is stored before asking, so Enter-to-keep is an informed choice
        print(f"\nSettings (saved in {CONFIG_PATH}):")
        print(f"  Word document : {cfg.get('docx') or '-'}")
        print(f"  Audio folder  : {cfg.get('audio_dir') or '-'}")
        print(f"  Output folder : {cfg.get('out_dir') or '<audio folder>/output'}")
        print(f"  Markers       : {cfg.get('granularity', 'word')}")
        print("  Enter keeps a value, '?' opens a picker, or type/drag a new path.\n")
    docx = Path(cfg["docx"]) if cfg.get("docx") else None
    audio = Path(cfg["audio_dir"]) if cfg.get("audio_dir") else None
    if force or docx is None or not docx.is_file():
        while True:
            docx = ask_path("Word document with the audio scripts", cfg.get("docx"), want_dir=False)
            if docx and docx.is_file() and docx.suffix.lower() == ".docx":
                break
            print("  not a .docx file — try again (Ctrl+C to quit)")
    if force or audio is None or not audio.is_dir():
        while True:
            audio = ask_path("Folder with its audio files", cfg.get("audio_dir"), want_dir=True)
            if audio and audio.is_dir():
                break
            print("  folder not found — try again")
    if force:
        cur_out = cfg.get("out_dir") or str(default_out_dir(audio))
        raw = input(f"Output folder [Enter = {cur_out}, '-' = <audio folder>/output]: ").strip()
        raw = re.sub(r"^&\s*", "", raw).strip().strip('"').strip("'")
        if raw == "-":
            cfg.pop("out_dir", None)
        elif raw:
            cfg["out_dir"] = raw
        g = input(f"Markers after every word or only at sentence ends? [w/s, Enter = "
                  f"{cfg.get('granularity', 'word')[0]}]: ").strip().lower()
        if g in ("w", "s"):
            cfg["granularity"] = "word" if g == "w" else "sentence"
    cfg["docx"], cfg["audio_dir"] = str(docx), str(audio)
    save_config(cfg)
    return True


def interactive(args) -> int:
    if args.reset and CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
        print(f"{CONFIG_PATH.name} removed: every setting will be asked again.")
    cfg = load_config()
    print("== align_engine: Word document + audio folder ==   (Ctrl+C quits at any time)\n")
    setup_paths(cfg, force=args.settings)
    aligner = Aligner(args.device, args.alignment_model, args.language, args.batch_size)

    while True:
        docx, audio_dir = Path(cfg["docx"]), Path(cfg["audio_dir"])
        out_dir = Path(cfg["out_dir"]) if cfg.get("out_dir") else default_out_dir(audio_dir)
        granularity = cfg.get("granularity", "word")
        try:
            tracks = read_tracks(docx)
        except Exception as e:  # noqa: BLE001
            print(f"Could not read {docx}: {e}")
            setup_paths(cfg, force=True)
            continue
        match_audio(tracks, audio_dir)
        done = done_indices(tracks, out_dir)

        print(f"\nDocument : {docx.name}\nAudio    : {audio_dir}\nOutput   : {out_dir}"
              f"\nMarkers  : {granularity}\n{summary_line(tracks, done)}")
        spec = input("\nTracks to align (26 | 26-31 | 4,7-9 | all | new)   "
                     "L = list, S = settings, Q = quit: ").strip()
        low = spec.lower()
        if low in ("q", "quit", "exit"):
            return 0
        if low in ("l", "list"):
            print_tracks(tracks, done)
            continue
        if low in ("s", "settings"):
            setup_paths(cfg, force=True)
            continue
        if not spec:
            continue
        try:
            selected = parse_selection(spec, tracks, done)
        except ValueError as e:
            print(f"  {e}")
            continue
        align_tracks(selected, aligner, out_dir, granularity, overwrite=None)


def batch_docx(args) -> int:
    cfg = load_config()
    docx = args.docx or (Path(cfg["docx"]) if cfg.get("docx") else None)
    audio_dir = args.audio_dir or (Path(cfg["audio_dir"]) if cfg.get("audio_dir") else None)
    if docx is None or audio_dir is None:
        print("--docx and --audio_dir are needed (or run without arguments once to set them).")
        return 2
    # the saved output folder / marker type apply here too (only when the audio folder is the saved one)
    saved_out = cfg.get("out_dir") if not args.audio_dir or str(audio_dir) == cfg.get("audio_dir") else None
    out_dir = args.out_dir or (Path(saved_out) if saved_out else default_out_dir(audio_dir))
    granularity = args.granularity or cfg.get("granularity", "word")
    tracks = read_tracks(docx)
    match_audio(tracks, audio_dir)
    done = done_indices(tracks, out_dir)
    print(summary_line(tracks, done))
    if not args.tracks:
        print_tracks(tracks, done)
        return 0
    try:
        selected = parse_selection(args.tracks, tracks, done)
    except ValueError as e:
        print(e)
        return 2
    aligner = Aligner(args.device, args.alignment_model, args.language, args.batch_size)
    _, failed = align_tracks(selected, aligner, out_dir, granularity,
                             overwrite=args.overwrite, dry_run=args.dry_run)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Old mode: <stem>.mp3 + <stem>.txt/.docx pairs in one folder
# ---------------------------------------------------------------------------

def read_pair_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        tracks = read_tracks(path)
        if not tracks:
            raise ValueError(f"no audio-script table in {path.name}")
        return tracks[0].text
    return path.read_text(encoding="utf-8-sig")


def batch_pairs(args) -> int:
    share = args.share_dir or (SCRIPT_DIR / "share")
    audio_dir, text_dir = args.pair_audio_dir or share, args.text_dir or share
    out_dir = args.out_dir or (share / "output")
    for label, d in (("audio", audio_dir), ("text", text_dir)):
        if not d.is_dir():
            print(f"{label} folder not found: {d}")
            return 1
    pairs = []
    for a in sorted(audio_dir.iterdir()):
        if a.suffix.lower() not in AUDIO_EXTS:
            continue
        t = next((text_dir / f"{a.stem}{e}" for e in (".docx", ".txt")
                  if (text_dir / f"{a.stem}{e}").exists()), None)
        if t is None:
            print(f"  [skip] no {a.stem}.docx or {a.stem}.txt")
        else:
            pairs.append((a, t))
    if not pairs:
        print("No (audio, text) pairs found.")
        return 1
    aligner = Aligner(args.device, args.alignment_model, args.language, args.batch_size)
    failed = []
    for a, t in pairs:
        print(f"Aligning {a.name} ...")
        try:
            text = read_pair_text(t)
            low = write_outputs(out_dir, a.stem, a.name, text, aligner.align(a, text), args.granularity or "word")
            print(f"  -> {a.stem}_draft.json ({low} low-confidence)")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAILED] {e}")
            failed.append(a.name)
    print(f"\nDone: {len(pairs) - len(failed)} succeeded, {len(failed)} failed.")
    return 1 if failed else 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", type=Path, help="Word document with all the audio-script tables")
    ap.add_argument("--audio_dir", type=Path, help="folder with the book's audio files")
    ap.add_argument("--tracks", help='which tracks: "26", "26-31", "4,7-9", "all", "new" (not yet aligned)')
    ap.add_argument("--out_dir", type=Path, help="default: <audio_dir>/output")
    ap.add_argument("--overwrite", action="store_true", help="re-align tracks that already have outputs")
    ap.add_argument("--dry-run", action="store_true", help="only show what would be aligned")
    ap.add_argument("--pairs", action="store_true", help="old mode: same-name audio + txt/docx pairs")
    ap.add_argument("--share_dir", type=Path, help="old mode: folder with the pairs (default: ./share)")
    ap.add_argument("--text_dir", type=Path, help="old mode: separate folder for the texts")
    ap.add_argument("--pair_audio_dir", type=Path, help="old mode: separate folder for the audio")
    ap.add_argument("--granularity", choices=["word", "sentence"], default=None,
                    help="word (default) or sentence; without it the saved setting is used")
    ap.add_argument("--settings", action="store_true",
                    help="open the settings questions first (change the saved document/folders/markers)")
    ap.add_argument("--reset", action="store_true",
                    help="forget align_config.json and ask every setting again")
    ap.add_argument("--language", default="eng", help="ISO 639-3 code (default: eng)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: automatic)")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="30-second pieces per model run (default: 2 on CPU, 8 on GPU; lowered "
                         "automatically if memory runs out)")
    ap.add_argument("--alignment_model", default=DEFAULT_MODEL,
                    help="HF model id or local path (default model is CC-BY-NC-4.0, see GUIDE.md)")
    args = ap.parse_args()

    try:
        if args.pairs or args.share_dir or args.text_dir or args.pair_audio_dir:
            return batch_pairs(args)
        if (args.docx or args.audio_dir or args.tracks) and not (args.settings or args.reset):
            return batch_docx(args)
        return interactive(args)
    except KeyboardInterrupt:
        print("\nBye.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
