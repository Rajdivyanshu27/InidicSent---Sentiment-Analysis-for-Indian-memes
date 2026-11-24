import difflib
import os
import re
import subprocess
import tempfile
import cv2
import numpy as np

PS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_run.ps1")


def _line_quality_score(text: str) -> float:
    """Score extracted text by clean alphanumeric word yield minus symbol noise."""
    if not text or not text.strip():
        return -1.0
    words = re.findall(r"[A-Za-z]{2,}", text)
    digits = re.findall(r"\d+", text)
    junk = len(re.findall(r"[^\w\s.,!?':;/\-&@]", text))
    alpha_chars = sum(len(w) for w in words)
    return alpha_chars + 1.8 * len(words) + 0.5 * len(digits) - 3.5 * junk


def _single_line_score(line: str) -> float:
    words = re.findall(r"[A-Za-z]{2,}", line)
    junk = len(re.findall(r"[^\w\s.,!?':;/\-&@]", line))
    weird_case = len(re.findall(r"[a-z][A-Z][a-z]|[A-Za-z]\d[A-Za-z]", line))
    return sum(len(w) for w in words) + len(words) - 3.5 * junk - 2.0 * weird_case


def _prepare_variants(image_path: str, tmp_dir: str) -> list[str]:
    """Create upscaled and contrast-tuned meme variants for multi-pass Windows OCR."""
    paths = [image_path]
    img = cv2.imread(image_path)
    if img is None:
        return paths

    h, w = img.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.35
    up = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    up_pad = cv2.copyMakeBorder(up, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Variant 1: Clean 2x padded RGB (preserves outlines without threshold artifacts)
    p1 = os.path.join(tmp_dir, "v_up.png")
    cv2.imwrite(p1, up_pad)
    paths.append(p1)

    # Variant 2: Mild unsharp mask RGB
    blur = cv2.GaussianBlur(up_pad, (0, 0), 1.1)
    sharp = cv2.addWeighted(up_pad, 1.35, blur, -0.35, 0)
    p2 = os.path.join(tmp_dir, "v_sharp.png")
    cv2.imwrite(p2, sharp)
    paths.append(p2)

    # Variant 3: Bright white-caption mask (catches light text on busy backgrounds)
    gray = cv2.cvtColor(up_pad, cv2.COLOR_BGR2GRAY)
    white_mask = np.where(gray > 195, 0, 255).astype(np.uint8)
    p3 = os.path.join(tmp_dir, "v_white.png")
    cv2.imwrite(p3, white_mask)
    paths.append(p3)

    return paths


def _is_similar(a: str, b: str) -> bool:
    na = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    nb = re.sub(r"[^a-z0-9]+", " ", b.lower()).strip()
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.62


def _merge_best_ocr(variants_out: list[str]) -> str:
    """Pick the strongest OCR pass and merge distinct missed lines without duplicates."""
    scored = []
    for idx, block in enumerate(variants_out):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        text = "\n".join(lines)
        # Small bonus (+2) for clean upscaled RGB (idx==1) or original (idx==0)
        bonus = 2.5 if idx in (0, 1) else 0.0
        scored.append((_line_quality_score(text) + bonus, lines))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return ""

    best_lines = list(scored[0][1])

    for _, lines in scored[1:]:
        for ln in lines:
            words = re.findall(r"[A-Za-z]{2,}", ln)
            junk = len(re.findall(r"[^\w\s.,!?':;/\-&@]", ln))
            if not words or junk > 1 or _single_line_score(ln) < 4:
                continue

            matched_idx = None
            for i, ex in enumerate(best_lines):
                if _is_similar(ln, ex):
                    matched_idx = i
                    break

            if matched_idx is None:
                best_lines.append(ln)
            elif _single_line_score(ln) > _single_line_score(best_lines[matched_idx]) + 2.5:
                # Replace corrupted line with significantly cleaner match from another pass
                best_lines[matched_idx] = ln

    return "\n".join(best_lines).strip()


def _extract_windows_ocr(variant_paths: list[str]) -> list[str] | None:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            PS_SCRIPT,
            "-ImagePaths",
            "|".join(variant_paths),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout.split("---OCR_VARIANT_SPLIT---")
    return None


def _extract_tesseract_ocr(variant_paths: list[str]) -> list[str]:
    """Linux/macOS fallback using tesseract CLI on each preprocessed variant."""
    blocks = []
    for path in variant_paths:
        try:
            proc = subprocess.run(
                ["tesseract", path, "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12,
            )
            blocks.append(proc.stdout if proc.returncode == 0 else "")
        except Exception:
            blocks.append("")
    return blocks


def extract_ocr_text(image_path: str) -> str:
    """Extract text from a meme image using multi-pass OCR (Windows Media OCR or Tesseract)."""
    if not os.path.isfile(image_path):
        return ""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            variant_paths = _prepare_variants(image_path, tmp_dir)
            raw_blocks = None
            if os.name == "nt" and os.path.isfile(PS_SCRIPT):
                raw_blocks = _extract_windows_ocr(variant_paths)
            if not raw_blocks:
                raw_blocks = _extract_tesseract_ocr(variant_paths)
            if raw_blocks:
                merged = _merge_best_ocr(raw_blocks)
                if merged:
                    return merged
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    sample = r"c:\Users\Asus\Downloads\Memotion 3-20261124T172055Z-1-001\Memotion 3\valImages\0.jpg"
    print("OCR output:\n" + extract_ocr_text(sample))
