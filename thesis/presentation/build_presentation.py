#!/usr/bin/env python3
"""Build the 10-minute thesis defense deck on top of the IMC house template.

Fills the supplied template (Cambria/Arial, orange FF9E1B accent) with the
agreed content: 6x6 bullets, a State-of-the-Art comparison table, three thesis
figures, and full speaker scripts in the notes pane. Re-runnable from the repo.
"""
from pathlib import Path
import fitz
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

ORANGE = RGBColor(0xFF, 0x9E, 0x1B)
GRAY = RGBColor(0x76, 0x86, 0x92)
LIGHT = RGBColor(0xFF, 0xE3, 0xC1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x26, 0x2B, 0x2F)

# ---------------------------------------------------------------- figures ----
def rasterize():
    jobs = {
        "pipeline": FIG / "fig_pipeline_overview.pdf",
        "speedup": FIG / "fig1_speedup_bar.pdf",
        "efficiency": FIG / "fig3_efficiency.pdf",
    }
    out = {}
    for name, pdf in jobs.items():
        png = ASSETS / f"{name}.png"
        doc = fitz.open(pdf)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
        pix.save(png)
        out[name] = (str(png), pix.width / pix.height)
    return out

# ---------------------------------------------------------------- helpers ----
def content_ph(slide):
    for sh in slide.shapes:
        if sh.is_placeholder and "Content" in sh.name:
            return sh
    return None

def title_ph(slide):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.name.startswith("Title"):
            return sh
    return None

def set_title(slide, text):
    t = title_ph(slide)
    if t:
        t.text_frame.text = text

def fill(ph, items, head=20, sub=18):
    """items: list of (text, level, bold, color|None). Header rows colored orange."""
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for text, level, bold, color in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(4)
        if level == 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = text
        run.font.size = Pt(head if level == 0 else sub)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color

def add_image_fit(slide, png, ar, box):
    """Fit image inside box (l,t,w,h inches) preserving aspect ratio, centered."""
    l, t, w, h = box
    if w / h > ar:           # box wider than image -> height limited
        ih = h; iw = h * ar
    else:                    # width limited
        iw = w; ih = w / ar
    il = l + (w - iw) / 2
    it = t + (h - ih) / 2
    return slide.shapes.add_picture(png, Inches(il), Inches(it),
                                    Inches(iw), Inches(ih))

def add_caption(slide, text, l, t, w, color=GRAY, size=12, italic=True):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(0.4))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.italic = italic; r.font.color.rgb = color
    return box

def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

# ---------------------------------------------------------------- build ------
def build():
    figs = rasterize()
    prs = Presentation(str(HERE / "template.pptx"))
    s = prs.slides

    # ---- Slide 2: Outline (already good; tighten wording) ----
    fill(content_ph(s[1]), [
        ("Motivation & Research Questions", 0, False, None),
        ("State of the Art", 0, False, None),
        ("Model — The Fingerprint", 0, False, None),
        ("Implementation", 0, False, None),
        ("Results", 0, False, None),
        ("Conclusion", 0, False, None),
    ], head=24)
    notes(s[1], "30 seconds. Six sections, ten minutes. The contribution is the "
          "caching framework; sleep staging is just the test vehicle.")

    # ---- Slide 3: Motivation ----
    fill(content_ph(s[2]), [
        ("Context", 0, True, ORANGE),
        ("EEG sleep staging — 128 subjects", 1, False, None),
        ("LOSO: one model per held-out subject", 1, False, None),
        ("The Challenge", 0, True, ORANGE),
        ("Per model: 128 subjects → 128 retrains", 1, False, None),
        ("18 model configs ≈ 21 hours", 1, False, None),
        ("Change one parameter → recompute all", 1, False, None),
        ("Framing", 0, True, ORANGE),
        ("Sleep EEG = vehicle, not the goal", 1, False, None),
    ])
    notes(s[2],
        "This began as a sleep-staging project. The standard protocol is "
        "Leave-One-Subject-Out: train on 127 people, test on the one held out, "
        "repeat for all 128 — so per model, 128 retrains. Run 18 model "
        "configurations and you're at ~21 hours per sweep, mostly recomputing "
        "byte-for-byte identical models. The problem was never accuracy; it was "
        "that changing one parameter threw all of it away. And to be clear: sleep "
        "data is just the vehicle. The contribution is the caching framework — "
        "LOSO is the hard, general case I use to prove it. (~1:40)")

    # ---- Slide 4: State of the Art (table + citation) ----
    set_title(s[3], "State of the Art")
    cp = content_ph(s[3])
    fill(cp, [("Every ingredient already exists — nobody combines them.",
               0, True, None)], head=20)
    cp.top = Inches(1.55); cp.height = Inches(0.6)
    rows = [
        ("Capability", "Tools"),
        ("Experiment tracking", "MLflow, W&B"),
        ("Pipeline caching", "DVC, ZenML"),
        ("Config fingerprinting", "DVC (MD5), ZenML (SHA-256)"),
        ("LOSO splitting", "scikit-learn"),
        ("Combination + held-out subject", "— none —"),
    ]
    tw, th = Inches(11.5), Inches(3.1)
    gf = s[3].shapes.add_table(len(rows), 2, Inches(0.9), Inches(2.35), tw, th)
    tbl = gf.table
    tbl.columns[0].width = Inches(5.2)
    tbl.columns[1].width = Inches(6.3)
    tbl.first_row = False
    for ri, (a, b) in enumerate(rows):
        for ci, txt in enumerate((a, b)):
            cell = tbl.cell(ri, ci)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_top = Pt(3); cell.margin_bottom = Pt(3)
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; r = p.add_run(); r.text = txt
            r.font.name = "Arial"; r.font.size = Pt(17)
            if ri == 0:                                   # header
                cell.fill.solid(); cell.fill.fore_color.rgb = ORANGE
                r.font.bold = True; r.font.color.rgb = WHITE
            elif ri == len(rows) - 1:                     # punch row
                cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT
                r.font.bold = True
                r.font.color.rgb = ORANGE if ci == 1 else DARK
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
                r.font.color.rgb = DARK
    # IEEE citation textbox already on slide
    for sh in s[3].shapes:
        if sh.shape_type == 17 and "TextBox" in sh.name:
            sh.text_frame.text = ("D. Kreuzberger, N. Kühl, S. Hirschl, "
                "“Machine Learning Operations (MLOps): Overview, Definition, "
                "and Architecture,” IEEE Access, vol. 11, 2023.")
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11); r.font.italic = True; r.font.color.rgb = GRAY
    notes(s[3],
        "Every ingredient already exists — tracking, caching, fingerprinting, "
        "cross-validation. What no tool does is the bottom row: combine them AND "
        "put the held-out subject into the cache key. Without that, a cache hit can "
        "silently hand you a model trained on the wrong people — no error, just "
        "an invalid result. That's the gap I close. (~1:00)")

    # ---- Slide 5: Model ----
    set_title(s[4], "Model — The Fingerprint")
    cp = content_ph(s[4])
    cp.left = Inches(0.75); cp.top = Inches(1.7)
    cp.width = Inches(11.8); cp.height = Inches(3.9)
    fill(cp, [
        ("Idea: every computation gets an identity", 0, True, ORANGE),
        ("Fingerprint = SHA-256(all inputs)", 1, False, None),
        ("Inputs: seed, code, params, features", 1, False, None),
        ("Plus the held-out subject ID", 1, False, None),
        ("Behaviour", 0, True, ORANGE),
        ("Same config → same key → reuse", 1, False, None),
        ("Change anything → new key → recompute", 1, False, None),
    ])
    fbox = s[4].shapes.add_textbox(Inches(0.9), Inches(5.95), Inches(11.5), Inches(0.9))
    fbox.fill.solid(); fbox.fill.fore_color.rgb = RGBColor(0xF2, 0xF3, 0xF4)
    fbox.line.color.rgb = ORANGE; fbox.line.width = Pt(1)
    tf = fbox.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    p = tf.paragraphs[0]
    r1 = p.add_run(); r1.text = "cache_key = SHA-256(seed, code, params, features) + "
    r1.font.name = "Consolas"; r1.font.size = Pt(16); r1.font.color.rgb = DARK
    r2 = p.add_run(); r2.text = "held_out_subject"
    r2.font.name = "Consolas"; r2.font.size = Pt(16); r2.font.bold = True
    r2.font.color.rgb = ORANGE
    notes(s[4],
        "The mechanism is one idea: give every computation an identity. I take the "
        "full configuration — seed, code version, hyperparameters, feature set "
        "— serialize it to canonical JSON, and hash it with SHA-256. Same inputs "
        "always give the same key; change one thing and the key changes, forcing a "
        "recompute. The decisive part, and the reason generic tools can't do this, is "
        "adding the held-out subject ID into the hash. That makes fold isolation "
        "structural, not a convention: a model trained with subject A out cannot be "
        "retrieved when subject B is the test subject. (Slide 2 was the problem; this "
        "is the one idea that fixes it.) (~1:30)")

    # ---- Slide 6: Implementation (left bullets + pipeline figure) ----
    set_title(s[5], "Implementation")
    cp = content_ph(s[5])
    cp.left = Inches(0.75); cp.top = Inches(1.7)
    cp.width = Inches(4.6); cp.height = Inches(4.8)
    fill(cp, [
        ("Two-layer cache", 0, True, ORANGE),
        ("Layer 1: features — always valid", 1, False, None),
        ("Layer 2: models — fingerprint-keyed", 1, False, None),
        ("Flow", 0, True, ORANGE),
        ("Miss → train → store → reuse", 1, False, None),
        ("Validated: zero hash collisions", 1, False, None),
    ], head=18, sub=16)
    png, ar = figs["pipeline"]
    add_image_fit(s[5], png, ar, (5.55, 1.75, 7.1, 4.4))
    add_caption(s[5], "LOSO experiment loop: fingerprint → cache check → "
                "train only on miss", 5.55, 6.2, 7.1)
    notes(s[5],
        "Here's the whole pipeline — and this diagram is the method on one "
        "slide. Two cache layers. Layer 1 holds the extracted features; they depend "
        "only on the raw signal, so they're computed once and stay valid. Layer 2 is "
        "the fingerprint-keyed model cache — that's where the speedup lives. The "
        "inner loop runs 128 times per configuration: fingerprint, check cache, train "
        "only on a miss. The outer loop sweeps all 18 configurations, and "
        "configurations that share folds reuse each other's cached models "
        "automatically. I verified it with deliberate invalidation tests — change "
        "a hyperparameter, a feature count, a subject — each produces a distinct "
        "fingerprint and the correct miss. Zero collisions. (~1:30)")

    # ---- Slide 7: Results a (Speed & Trust) ----
    set_title(s[6], "Results — Speed & Trust")
    cp = content_ph(s[6])
    cp.left = Inches(0.75); cp.top = Inches(1.7)
    cp.width = Inches(4.9); cp.height = Inches(4.8)
    fill(cp, [
        ("Speed", 0, True, ORANGE),
        ("XGBoost: median 54×, peak 202×", 1, False, None),
        ("3,816 s → 18.9 s", 1, False, None),
        ("Feature cache: 227×", 1, False, None),
        ("Trust", 0, True, ORANGE),
        ("100% hit rate — 2,304 / 2,304", 1, False, None),
        ("100% identical predictions", 1, False, None),
        ("85.5% accuracy, κ = 0.74", 1, False, None),
    ], head=18, sub=16)
    png, ar = figs["speedup"]
    add_image_fit(s[6], png, ar, (5.85, 1.75, 6.8, 4.4))
    add_caption(s[6], "Cold vs. warm wall-clock per configuration (log scale)",
                5.85, 6.2, 6.8)
    notes(s[6],
        "The headline: for XGBoost, a median 54x speedup, and the best configuration "
        "went from a 64-minute cold run to 19 seconds — 202x. Feature extraction "
        "alone is 227x faster. Two numbers I want to state explicitly, because they're "
        "the scientific claims, not just speed: every one of the 2,304 warm operations "
        "was a hit — a 100% hit rate — and every cached model produced "
        "bitwise-identical predictions to retraining. And the cached models still "
        "reach 85.5% accuracy, kappa 0.74 — so this is faster AND provably the "
        "same result, at no cost to quality. (~1:30)")

    # ---- Slide 8: Results b (Viability & Scaling) ----
    set_title(s[7], "Results — Viability & Scaling")
    cp = content_ph(s[7])
    cp.left = Inches(0.75); cp.top = Inches(1.7)
    cp.width = Inches(4.9); cp.height = Inches(4.8)
    fill(cp, [
        ("Viability", 0, True, ORANGE),
        ("15 models: 11 viable, 4 not", 1, False, None),
        ("η (eta) = seconds saved / MB", 1, False, None),
        ("Above the line → worth caching", 1, False, None),
        ("Cost & Scaling", 0, True, ORANGE),
        ("XGBoost 1.5 MB (viable)", 1, False, None),
        ("RF 18.5 GB (not viable)", 1, False, None),
        ("SVM-RBF: 9× → 201× with scale", 1, False, None),
    ], head=18, sub=16)
    png, ar = figs["efficiency"]
    add_image_fit(s[7], png, ar, (5.85, 1.75, 6.8, 4.4))
    add_caption(s[7], "Compute saved vs. storage cost; dashed line = viability "
                "boundary", 5.85, 6.2, 6.8)
    notes(s[7],
        "Caching isn't free for every model, so I built a metric — eta, the same "
        "symbol physics uses for efficiency — seconds of compute saved per "
        "megabyte stored. I tested 15 models. Everything above this diagonal recovers "
        "more compute than it spends on storage I/O: 11 pass, 4 fail. The failures "
        "cluster bottom-right — Random Forest, Extra Trees, kNN — big files, "
        "little saved. And the scaling result is the general takeaway: SVM went from "
        "9x at ten subjects to 201x at 128. Caching gets MORE valuable as the study "
        "grows, not less. (~1:20)")

    # ---- Slide 9: Conclusion ----
    fill(content_ph(s[8]), [
        ("Contributions", 0, True, ORANGE),
        ("Fold-aware fingerprint → no contamination", 1, False, None),
        ("Order-of-magnitude speedup, reproducible", 1, False, None),
        ("Bonus: every fold's model preserved", 1, False, None),
        ("Takeaways", 0, True, ORANGE),
        ("Caching scales with study size", 1, False, None),
        ("Method transfers — the numbers don't", 1, False, None),
        ("RQ1–RQ4: answered", 1, False, None),
    ])
    notes(s[8],
        "To close: the fingerprint makes fold-safe caching structural, the speedups "
        "are order-of-magnitude, and reproducibility is guaranteed by construction "
        "— that answers all four research questions: efficiency, reproducibility, "
        "scalability, and the storage trade-off. There's also a second win beyond "
        "speed: because I persist every fold's model, not just the metrics, a clinician "
        "can flag an outlier subject and an ML specialist can pull that subject's exact "
        "model and inspect its feature importances — for free. Model evaluation "
        "meeting domain knowledge. The honest caveat: the specific numbers depend on "
        "hardware and model choice — but the method, the fingerprint and the eta "
        "metric, transfers to any subject-level cross-validation domain. (~1:20)")

    # ---- Slide 10: Thank you (leave content; add note) ----
    notes(s[9], "Thank you — I'm happy to take questions. Advance to backup "
          "slides as needed (full results table, eta ranking, confusion matrix, "
          "global-vs-fold leakage).")

    # ---- Slide 11: Backup ----
    fill(content_ph(s[10]), [
        ("Backup material", 0, True, ORANGE),
        ("Full 9-config results table (XGB + RF)", 1, False, None),
        ("η ranking — all 15 models", 1, False, None),
        ("Confusion matrix + per-class F1 (N1)", 1, False, None),
        ("Global vs. per-fold selection (≤1.1%)", 1, False, None),
    ])
    notes(s[10], "Optional. Pull these up only if asked. The global-vs-fold slide "
          "directly addresses the train-set leakage point.")

    out = HERE / "Thesis_Presentation_Gorzel.pptx"
    prs.save(str(out))
    print("saved:", out)
    return out

if __name__ == "__main__":
    build()
