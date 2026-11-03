import io
import pandas as pd


class LifeLensPDF:
    """
    Self-contained, dependency-free PDF 1.4 document generator.
    Produces high-fidelity, printable PDF reports for LifeLens.
    """

    def __init__(self, title="LifeLens Wellness Report"):
        self.title = title
        self.pages = []
        self.current_page_ops = []
        self.width = 612   # Standard Letter width
        self.height = 792  # Standard Letter height
        self.margin = 38
        self.cursor_y = self.height - self.margin

    def add_page(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []
        self.cursor_y = self.height - self.margin
        # Subtle top & bottom dividing rules
        self.draw_rect(self.margin, self.height - 24, self.width - 2 * self.margin, 0.75, fill_color=(0.82, 0.78, 0.90))
        self.draw_rect(self.margin, 34, self.width - 2 * self.margin, 0.75, fill_color=(0.82, 0.78, 0.90))

    def check_space(self, needed):
        if self.cursor_y - needed < 48:
            self.add_page()

    def draw_rect(self, x, y, w, h, fill_color=None, stroke_color=None, stroke_width=1):
        op = []
        if fill_color:
            op.append(f"{fill_color[0]:.3f} {fill_color[1]:.3f} {fill_color[2]:.3f} rg")
        if stroke_color:
            op.append(f"{stroke_color[0]:.3f} {stroke_color[1]:.3f} {stroke_color[2]:.3f} RG")
            op.append(f"{stroke_width} w")
        op.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re")
        if fill_color and stroke_color:
            op.append("B")
        elif fill_color:
            op.append("f")
        elif stroke_color:
            op.append("S")
        self.current_page_ops.append(" ".join(op))

    def draw_text(self, text, x, y, font="Helvetica", size=10, color=(0.16, 0.14, 0.23)):
        clean_text = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font_ref = "/F2" if "Bold" in font else "/F1"
        r, g, b = color
        op = f"BT {r:.3f} {g:.3f} {b:.3f} rg {font_ref} {size} Tf {x:.1f} {y:.1f} Td ({clean_text}) Tj ET"
        self.current_page_ops.append(op)

    def add_heading(self, text, size=15, space_after=8, color=(0.47, 0.31, 0.78)):
        self.check_space(size + space_after + 8)
        self.cursor_y -= (size + 3)
        self.draw_text(text, self.margin, self.cursor_y, font="Helvetica-Bold", size=size, color=color)
        self.cursor_y -= space_after

    def add_subheading(self, text, size=11, space_after=6, color=(0.18, 0.15, 0.25)):
        self.check_space(size + space_after + 6)
        self.cursor_y -= (size + 2)
        self.draw_text(text, self.margin, self.cursor_y, font="Helvetica-Bold", size=size, color=color)
        self.cursor_y -= space_after

    def add_paragraph(self, text, size=9, line_height=13.5, space_after=8, color=(0.28, 0.25, 0.35)):
        words = str(text).split()
        max_w = self.width - 2 * self.margin
        max_chars = int(max_w / (size * 0.52))

        lines = []
        curr_line = []
        curr_len = 0
        for w in words:
            if curr_len + len(w) + 1 > max_chars:
                lines.append(" ".join(curr_line))
                curr_line = [w]
                curr_len = len(w)
            else:
                curr_line.append(w)
                curr_len += len(w) + 1
        if curr_line:
            lines.append(" ".join(curr_line))

        needed = len(lines) * line_height + space_after
        self.check_space(needed)
        for line in lines:
            self.cursor_y -= line_height
            self.draw_text(line, self.margin, self.cursor_y, font="Helvetica", size=size, color=color)
        self.cursor_y -= space_after

    def add_card_table(self, title, items, cols=2):
        self.add_subheading(title, size=10.5, space_after=6)
        col_w = (self.width - 2 * self.margin) / cols
        row_h = 24
        needed = ((len(items) + cols - 1) // cols) * row_h + 8
        self.check_space(needed)

        for idx, (label, val) in enumerate(items):
            c_idx = idx % cols
            r_idx = idx // cols
            x = self.margin + c_idx * col_w
            y = self.cursor_y - (r_idx + 1) * row_h

            self.draw_rect(x, y + 2, col_w - 6, row_h - 4, fill_color=(0.97, 0.96, 0.99), stroke_color=(0.90, 0.86, 0.95))
            self.draw_text(label, x + 8, y + 8, font="Helvetica", size=8.5, color=(0.42, 0.38, 0.50))
            val_str = str(val)
            self.draw_text(val_str, x + col_w - 14 - (len(val_str) * 5.8), y + 8, font="Helvetica-Bold", size=9, color=(0.16, 0.14, 0.23))

        self.cursor_y -= (((len(items) + cols - 1) // cols) * row_h + 8)

    def render(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []

        total_pages = len(self.pages)
        for idx, page in enumerate(self.pages):
            p_num = f"Page {idx + 1} of {total_pages}"
            hdr = f"LifeLens Wellness Intelligence  |  {self.title}"
            clean_hdr = hdr.replace("(", "\\(").replace(")", "\\)")
            clean_p = p_num.replace("(", "\\(").replace(")", "\\)")
            page.append(f"BT 0.52 0.48 0.60 rg /F1 8 Tf {self.margin} {self.height - 18} Td ({clean_hdr}) Tj ET")
            page.append(f"BT 0.52 0.48 0.60 rg /F1 8 Tf {self.width - self.margin - 50} 22 Td ({clean_p}) Tj ET")
            page.append(f"BT 0.52 0.48 0.60 rg /F1 8 Tf {self.margin} 22 Td (Confidential Personal Health Record - For Personal Wellness Awareness Only) Tj ET")

        objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [] /Count {total_pages} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]

        page_refs = [f"{5 + i * 2} 0 R" for i in range(total_pages)]
        objects[1] = f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {total_pages} >>"

        next_obj_num = 5
        for i in range(total_pages):
            content_idx = next_obj_num + 1
            page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_idx} 0 R >>"
            content_str = "\n".join(self.pages[i])
            content_bytes = content_str.encode("latin1", errors="replace")
            content_obj = f"<< /Length {len(content_bytes)} >>\nstream\n{content_str}\nendstream"
            objects.append(page_obj)
            objects.append(content_obj)
            next_obj_num += 2

        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for idx, obj in enumerate(objects):
            offsets.append(buf.tell())
            buf.write(f"{idx + 1} 0 obj\n{obj}\nendobj\n".encode("latin1", errors="replace"))

        xref_offset = buf.tell()
        buf.write(f"xref\n0 {len(objects) + 1}\n".encode("latin1"))
        buf.write(b"0000000000 65535 f \n")
        for offset in offsets:
            buf.write(f"{offset:010d} 00000 n \n".encode("latin1"))

        trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        buf.write(trailer.encode("latin1"))
        return buf.getvalue()


def generate_wellness_pdf(report_type, stats_dict, period_str, ai_text="", recommendations=None):
    """
    Constructs a complete executive PDF report.
    Returns: PDF raw bytes
    """
    title = f"{report_type} Wellness Report"
    pdf = LifeLensPDF(title)
    pdf.add_page()

    # Brand Banner Box
    pdf.draw_rect(pdf.margin, pdf.cursor_y - 48, pdf.width - 2 * pdf.margin, 48, fill_color=(0.95, 0.93, 0.98), stroke_color=(0.85, 0.80, 0.94), stroke_width=1.2)
    pdf.draw_text("LifeLens  -  Personal Wellness Intelligence", pdf.margin + 12, pdf.cursor_y - 18, font="Helvetica-Bold", size=11, color=(0.47, 0.31, 0.78))
    pdf.draw_text(f"Executive {report_type} Report  |  {period_str}", pdf.margin + 12, pdf.cursor_y - 34, font="Helvetica", size=9, color=(0.40, 0.36, 0.48))
    pdf.cursor_y -= 58

    # Executive Score Banner
    start_sc = stats_dict.get("start_score", "N/A")
    end_sc = stats_dict.get("end_score", "N/A")
    delta_sc = stats_dict.get("score_delta", "±0")
    cat = stats_dict.get("category", "General")
    consistency = stats_dict.get("consistency", "N/A")

    pdf.add_card_table("Executive Wellness Score Summary", [
        ("Starting Wellness Score", str(start_sc)),
        ("Ending Wellness Score", str(end_sc)),
        ("Score Trajectory Change", str(delta_sc)),
        ("Wellness Classification", str(cat)),
        ("Check-in Consistency", str(consistency)),
        ("Total Recorded Days", f"{stats_dict.get('days_recorded', 0)} days"),
    ], cols=2)

    # Core Metrics Table
    pdf.add_card_table("Average Lifestyle Indicators", [
        ("Average Sleep Duration", f"{stats_dict.get('avg_sleep', 'N/A')}"),
        ("Average Mood Rating", f"{stats_dict.get('avg_mood', 'N/A')}"),
        ("Average Vital Energy", f"{stats_dict.get('avg_energy', 'N/A')}"),
        ("Average Stress Level", f"{stats_dict.get('avg_stress', 'N/A')}"),
        ("Average Productivity", f"{stats_dict.get('avg_productivity', 'N/A')}"),
        ("Daily Exercise Average", f"{stats_dict.get('avg_exercise', 'N/A')}"),
        ("Daily Screen Time", f"{stats_dict.get('avg_screen', 'N/A')}"),
        ("Daily Study / Work", f"{stats_dict.get('avg_study', 'N/A')}"),
    ], cols=2)

    # Best & Worst Days
    if stats_dict.get("best_day") or stats_dict.get("worst_day"):
        pdf.add_card_table("Peak & Challenging Days Observed", [
            ("Peak Wellness Day", f"{stats_dict.get('best_day', 'N/A')} (Score: {stats_dict.get('best_score', 'N/A')})"),
            ("Challenging Recovery Day", f"{stats_dict.get('worst_day', 'N/A')} (Score: {stats_dict.get('worst_score', 'N/A')})"),
        ], cols=2)

    # AI Observations Section
    if ai_text:
        pdf.add_subheading("AI Wellness Synthesis & Narrative Analysis", size=10.5)
        pdf.add_paragraph(ai_text)

    # Recommendations Section
    if recommendations:
        pdf.add_subheading("Targeted Wellness Recommendations & Habit Actions", size=10.5)
        for rec in recommendations[:4]:
            pdf.add_paragraph(f"- {rec}")

    # Health Disclaimer
    pdf.check_space(55)
    pdf.draw_rect(pdf.margin, pdf.cursor_y - 42, pdf.width - 2 * pdf.margin, 42, fill_color=(0.99, 0.98, 0.94), stroke_color=(0.95, 0.88, 0.72))
    pdf.draw_text("Important Health Notice & Disclaimer:", pdf.margin + 8, pdf.cursor_y - 14, font="Helvetica-Bold", size=7.5, color=(0.70, 0.45, 0.10))
    pdf.draw_text("LifeLens is an educational behavioral tracking tool, not a medical or clinical diagnosis system. The metrics and AI insights", pdf.margin + 8, pdf.cursor_y - 25, font="Helvetica", size=7, color=(0.45, 0.40, 0.30))
    pdf.draw_text("do not constitute medical advice. Consult qualified healthcare professionals for medical conditions.", pdf.margin + 8, pdf.cursor_y - 35, font="Helvetica", size=7, color=(0.45, 0.40, 0.30))
    pdf.cursor_y -= 50

    return pdf.render()

