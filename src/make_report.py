"""Builds report/report.pdf from report/narrative.md.
Syntax inside narrative.md:  '# H1', '## H2', '- bullet', blank-line separated paragraphs, **bold**,
  {{IMG:path|width_cm}}   embed an image
  {{TABLE:path.csv|col1,col2,...|max_rows}}   embed (part of) a CSV as a table
Warns about every leftover '[FILL' placeholder so you cannot forget one.
"""
import re, sys, os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem, PageBreak
from reportlab.lib.utils import ImageReader

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7, leading=8.5)
HEAD = ParagraphStyle("h", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")


def md_inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return re.sub(r"(https?://[^\s)]+)", r"<link href='\1' color='blue'>\1</link>", t)


def table(path, cols, max_rows):
    df = pd.read_csv(path)
    cols = [c for c in (cols or list(df.columns)) if c in df.columns]
    df = df[cols].head(max_rows)
    data = [[Paragraph(c, HEAD) for c in cols]] + [[Paragraph(md_inline(str(v)), CELL) for v in r] for r in df.values]
    w = (A4[0] - 3.2 * cm) / len(cols)
    t = Table(data, colWidths=[w] * len(cols), repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    return t


def main(src="report/narrative.md", out="report/report.pdf"):
    text = open(src, encoding="utf-8").read()
    left = len(re.findall(r"\[FILL", text))
    if left: print(f"WARNING: {left} '[FILL' placeholders still in {src}")
    story, para, bullets = [], [], []

    def flush():
        nonlocal para, bullets
        if para:
            story.append(Paragraph(md_inline(" ".join(para)), BODY)); story.append(Spacer(1, 5)); para = []
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(md_inline(b), BODY)) for b in bullets], bulletType="bullet", leftIndent=14))
            story.append(Spacer(1, 5)); bullets = []

    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"\{\{IMG:(.+?)(?:\|(.+?))?\}\}", s)
        t = re.match(r"\{\{TABLE:(.+?)(?:\|(.*?))?(?:\|(\d+))?\}\}", s)
        if m:
            flush()
            p, w = m.group(1), float(m.group(2) or 16)
            if not os.path.exists(p): story.append(Paragraph(f"<i>[missing image {p}]</i>", BODY)); print("missing", p); continue
            iw, ih = ImageReader(p).getSize()
            story += [Image(p, width=w * cm, height=w * cm * ih / iw), Spacer(1, 6)]
        elif t:
            flush()
            p = t.group(1)
            if not os.path.exists(p): story.append(Paragraph(f"<i>[missing table {p}]</i>", BODY)); print("missing", p); continue
            cols = [c.strip() for c in (t.group(2) or "").split(",") if c.strip()]
            story += [table(p, cols, int(t.group(3) or 60)), Spacer(1, 8)]
        elif s.startswith("# "):
            flush(); story.append(Paragraph(md_inline(s[2:]), ss["Heading1"]))
        elif s.startswith("## "):
            flush(); story.append(Paragraph(md_inline(s[3:]), ss["Heading2"]))
        elif s.startswith("- "):
            if para: flush()
            bullets.append(s[2:])
        elif s == "---":
            flush(); story.append(PageBreak())
        elif s == "":
            flush()
        else:
            para.append(s)
    flush()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="CV Intern Task - Egg segmentation").build(story)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
