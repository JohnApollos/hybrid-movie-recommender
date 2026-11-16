"""Generate publication-grade 16:9 presentation slides PDF matching specifications.

Specifications:
  - 16:9 Landscape format
  - Projector-safe light background / dark text
  - 12-slide outline
  - Professional typography, slide numbers, clean styling
"""
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "docs" / "Group04_MovieRecommender_Slides.pdf"

# 16:9 Landscape Page Size (width 792 pt, height 445.5 pt)
PAGE_WIDTH = 792.0
PAGE_HEIGHT = 445.5


class SlideCanvas(canvas.Canvas):
    """Draws consistent slide headers, footers, and page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_slide_chrome(num_pages)
            super().showPage()
        super().save()

    def draw_slide_chrome(self, total_pages):
        self.saveState()
        # Slide Footer
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 24, "Hybrid Movie Recommender &bull; JKUAT BSc Data Science & Analytics &bull; Group 04")
        self.drawRightString(PAGE_WIDTH - 40, 24, f"Slide {self._pageNumber} of {total_pages}")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.8)
        self.line(40, 36, PAGE_WIDTH - 40, 36)
        self.restoreState()


def build_slides():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    title_slide_title = ParagraphStyle(
        "TitleSlideTitle",
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12
    )

    title_slide_sub = ParagraphStyle(
        "TitleSlideSub",
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceAfter=20
    )

    slide_title = ParagraphStyle(
        "SlideTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4
    )

    slide_cat = ParagraphStyle(
        "SlideCat",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2563EB"),
        textTransform="uppercase",
        spaceAfter=4
    )

    bullet_body = ParagraphStyle(
        "BulletBody",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=8
    )

    cell_body = ParagraphStyle(
        "CellBody",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    cell_hdr = ParagraphStyle(
        "CellHdr",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    def make_header(category, title_text):
        elems = [
            Paragraph(category.upper(), slide_cat),
            Paragraph(title_text, slide_title),
            HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=3, spaceAfter=14)
        ]
        return elems

    # ---------------------------------------------------------------- SLIDE 1: TITLE
    story.append(Spacer(1, 40))
    story.append(Paragraph("JKUAT BSC DATA SCIENCE & ANALYTICS &bull; GROUP 04", ParagraphStyle("Cat1", fontName="Helvetica-Bold", fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#2563EB"), spaceAfter=12)))
    story.append(Paragraph("Hybrid Movie Recommendation System", title_slide_title))
    story.append(Paragraph("Collaborative Filtering, SVD Latent Factors, Content-Based TF-IDF, and PyTorch Deep Learning", title_slide_sub))
    story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=16))
    meta_box = [
        [Paragraph("<b>Authors:</b> Group 04 (15 Research Contributors)", cell_body), Paragraph("<b>Dataset:</b> MovieLens 100k (ml-latest-small)", cell_body)],
        [Paragraph("<b>Supervisor:</b> Dept. of Computing & Data Science", cell_body), Paragraph("<b>Deployment:</b> Streamlit Web App + PyTorch", cell_body)],
    ]
    t_meta = Table(meta_box, colWidths=[350, 350])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 2: PROBLEM & OBJECTIVES
    story.extend(make_header("Introduction & Motivation", "Problem Statement & Project Objectives"))
    col1 = [
        Paragraph("<b>Recommender System Challenges:</b>", bullet_body),
        Paragraph("&bull; <b>Extreme Sparsity:</b> 98.30% of matrix entries unobserved; users rate &lt;2% of movies.", bullet_body),
        Paragraph("&bull; <b>Cold-Start Barrier:</b> Collaborative filtering collapses on new users with zero history.", bullet_body),
        Paragraph("&bull; <b>Popularity Bias:</b> Heuristic models over-recommend blockbusters, creating monoculture.", bullet_body),
    ]
    col2 = [
        Paragraph("<b>Project Objectives:</b>", bullet_body),
        Paragraph("&bull; Implement and benchmark 8 models across collaborative, latent, and content paradigms.", bullet_body),
        Paragraph("&bull; Engineer a <b>Weighted Hybrid</b> optimizing validation ranking metrics.", bullet_body),
        Paragraph("&bull; Deploy an interactive Streamlit app with cold-start onboarding & live feedback logging.", bullet_body),
    ]
    t_s2 = Table([[col1, col2]], colWidths=[350, 350])
    t_s2.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_s2)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 3: DATASET & EDA
    story.extend(make_header("Data Engineering", "Dataset Architecture & Exploratory Insights"))
    d_col1 = [
        Paragraph("<b>MovieLens 100k Dimensions:</b>", bullet_body),
        Paragraph("&bull; <b>100,836 ratings</b> &bull; <b>9,742 movies</b> &bull; <b>610 users</b>.", bullet_body),
        Paragraph("&bull; <b>Matrix Sparsity:</b> 98.30% sparse (1.70% interaction density).", bullet_body),
        Paragraph("&bull; <b>Rating Scale:</b> 0.5 to 5.0 stars &bull; Median = 3.50 ★.", bullet_body),
        Paragraph("&bull; <b>Relevance Cutoff:</b> <b>48.2% of ratings &ge; 4.0</b> (ground-truth target).", bullet_body),
    ]
    d_col2 = [
        Paragraph("<b>Empirical Consumption Patterns:</b>", bullet_body),
        Paragraph("&bull; <b>Severe Long-Tail:</b> Top 100 movies hold 16.1% of all ratings.", bullet_body),
        Paragraph("&bull; <b>Sparse Inventory:</b> 62.5% of movies have fewer than 5 ratings.", bullet_body),
        Paragraph("&bull; <b>User Activity Skew:</b> Median user has 70.5 ratings (max = 2,698).", bullet_body),
        Paragraph("&bull; <b>Metadata:</b> 99.7% of movies have genre tags (avg 2.27 genres).", bullet_body),
    ]
    t_s3 = Table([[d_col1, d_col2]], colWidths=[350, 350])
    t_s3.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_s3)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 4: SYSTEM ARCHITECTURE
    story.extend(make_header("Engineering Design", "End-to-End System Architecture & Data Pipeline"))
    arch_box = [
        [Paragraph("<b>Phase 1: Ingestion & Splits</b>", cell_hdr), Paragraph("<b>Phase 2: Model Engines</b>", cell_hdr), Paragraph("<b>Phase 3: Hybrid & Serving</b>", cell_hdr)],
        [
            Paragraph("MovieLens 100k Data<br/>&bull; Chronological LOO Split (610 test rows, Rel &ge; 4.0)<br/>&bull; Random 80/20 RMSE Split (20k test)", cell_body),
            Paragraph("&bull; <b>User-CF:</b> Mean-centered Cosine<br/>&bull; <b>Item-CF:</b> Adjusted Cosine<br/>&bull; <b>SVD:</b> 100 Latent Factors<br/>&bull; <b>Content:</b> TF-IDF (Genres+Tags)<br/>&bull; <b>NCF:</b> PyTorch NeuMF (GMF+MLP)", cell_body),
            Paragraph("&bull; <b>Z-Score Normalization</b><br/>&bull; <b>Validation Weight Grid Search</b><br/>&bull; <b>Streamlit Web Deployment</b><br/>&bull; <b>Persistent Feedback Logging</b>", cell_body)
        ]
    ]
    t_arch = Table(arch_box, colWidths=[230, 250, 230])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC")])
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 14))
    story.append(Paragraph("Decoupled contract interface: every squad writes standard top-10 CSVs (<code>userId, movieId, rank, score</code>), consumed directly by the hybrid and Streamlit app.", bullet_body))
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 5: CLASSICAL MODELS (CF + SVD)
    story.extend(make_header("Algorithmic Formulations", "Classical Collaborative Filtering & SVD Matrix Factorization"))
    cf_col1 = [
        Paragraph("<b>User-Based CF (Cosine):</b>", bullet_body),
        Paragraph("&bull; Mean-centered cosine similarity on $R^c = (R - \\bar{r}_u) \\cdot B$.", bullet_body),
        Paragraph("&bull; Retains top $k=50$ positive neighbors.", bullet_body),
        Paragraph("&bull; Similarity shrinkage $\\beta=5.0$ damps noisy small overlaps.", bullet_body),
        Paragraph("&bull; Strong personalized ranking: <b>Recall@10 = 0.0909</b>.", bullet_body),
    ]
    cf_col2 = [
        Paragraph("<b>Item-Based CF & SVD Factorization:</b>", bullet_body),
        Paragraph("&bull; <b>Item-CF:</b> Adjusted cosine similarity with co-rater shrinkage ($\\lambda=10.0$); stable neighborhoods.", bullet_body),
        Paragraph("&bull; <b>SVD:</b> 100 latent factors with user/item biases learned via regularized SGD over 20 epochs.", bullet_body),
        Paragraph("&bull; <b>Key Result:</b> SVD achieved the lowest prediction error: <b>RMSE = 0.8831</b>.", bullet_body),
    ]
    t_cf = Table([[cf_col1, cf_col2]], colWidths=[350, 350])
    t_cf.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_cf)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 6: CONTENT & HYBRID (CORE)
    story.extend(make_header("Core Deliverable", "Content-Based Filtering & The Weighted Hybrid Engine"))
    hy_col1 = [
        Paragraph("<b>Content-Based TF-IDF Engine:</b>", bullet_body),
        Paragraph("&bull; Combines pipe-delimited genres and user-applied tags into sublinear TF-IDF vectors.", bullet_body),
        Paragraph("&bull; User profile = rating-weighted centroid vector.", bullet_body),
        Paragraph("&bull; Highest catalog exploration (<b>21.68% coverage</b>); acts as the instant cold-start fallback engine.", bullet_body),
    ]
    hy_col2 = [
        Paragraph("<b>Weighted Hybrid Strategy:</b>", bullet_body),
        Paragraph("&bull; Standardizes candidate model scores via per-user z-scores, eliminating arbitrary scale dominance.", bullet_body),
        Paragraph("&bull; Leakage-free grid search on validation split ($N_{\\text{val}}=5$): optimal weights <b>$w_{\\text{user}}=0.60, w_{\\text{item}}=0.40$</b>.", bullet_body),
        Paragraph("&bull; <b>State-of-the-Art:</b> Highest <b>Recall@10 = 0.0964</b> and <b>NDCG@10 = 0.0521</b>.", bullet_body),
    ]
    t_hy = Table([[hy_col1, hy_col2]], colWidths=[350, 350])
    t_hy.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_hy)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 7: DEEP LEARNING (NCF)
    story.extend(make_header("Topic 3 Requirement", "Deep Learning: Neural Collaborative Filtering (NCF / NeuMF)"))
    ncf_col1 = [
        Paragraph("<b>NeuMF Architecture (PyTorch):</b>", bullet_body),
        Paragraph(r"&bull; <b>GMF Branch:</b> Computes linear element-wise latent product ($\mathbf{p}_u \odot \mathbf{q}_i$).", bullet_body),
        Paragraph("&bull; <b>MLP Branch:</b> 3 dense layers learning non-linear feature interactions with ReLU activations.", bullet_body),
        Paragraph("&bull; <b>Objective:</b> Binary Cross-Entropy loss with 4 negative samples per positive interaction.", bullet_body),
    ]
    ncf_col2 = [
        Paragraph("<b>Hyperparameter Sweep & Insights:</b>", bullet_body),
        Paragraph("&bull; Swept 7 configurations across latent dimensions and dropout rates.", bullet_body),
        Paragraph("&bull; Compact architecture (dim=16, layers 32-16-8, 10% dropout) prevented overfitting on sparse data.", bullet_body),
        Paragraph("&bull; Reached <b>Val Recall = 0.0824</b> at epoch 16; final test Recall@10 = <b>0.0744</b>.", bullet_body),
    ]
    t_ncf = Table([[ncf_col1, ncf_col2]], colWidths=[350, 350])
    t_ncf.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_ncf)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 8: EVALUATION PROTOCOL
    story.extend(make_header("Scientific Rigor", "Evaluation Protocol & Performance Metrics"))
    proto_col1 = [
        Paragraph("<b>Leave-One-Out (LOO) Ranking Protocol:</b>", bullet_body),
        Paragraph("&bull; Holds out each user's latest interaction as test item.", bullet_body),
        Paragraph("&bull; <b>Strict Relevance:</b> True rating must be <b>&ge; 4.0</b> (363 relevant evaluation users). Recommending disliked films is penalized.", bullet_body),
        Paragraph("&bull; <b>80/20 Random Split:</b> Independent 20k test ratings for rating prediction error (RMSE).", bullet_body),
    ]
    proto_col2 = [
        Paragraph("<b>Evaluation Metric Suite:</b>", bullet_body),
        Paragraph("&bull; <b>Recall@10:</b> Fraction of relevant held-out items recovered.", bullet_body),
        Paragraph("&bull; <b>Precision@10:</b> Proportion of top-10 recs that are hits.", bullet_body),
        Paragraph("&bull; <b>NDCG@10:</b> Logarithmic ranking discount quality.", bullet_body),
        Paragraph("&bull; <b>RMSE:</b> Explicit rating prediction squared error.", bullet_body),
        Paragraph("&bull; <b>Catalog Coverage:</b> % of total 9,742 catalog recommended.", bullet_body),
    ]
    t_proto = Table([[proto_col1, proto_col2]], colWidths=[350, 350])
    t_proto.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_proto)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 9: RESULTS VS BASELINES
    story.extend(make_header("Empirical Benchmarks", "Official Experimental Results vs. Baselines"))
    res_table = [
        [Paragraph("<b>Model</b>", cell_hdr), Paragraph("<b>P@10</b>", cell_hdr), Paragraph("<b>R@10</b>", cell_hdr), Paragraph("<b>NDCG@10</b>", cell_hdr), Paragraph("<b>RMSE</b>", cell_hdr), Paragraph("<b>Catalog Cov.</b>", cell_hdr)],
        [Paragraph("Popularity Baseline", cell_body), Paragraph("0.0063", cell_body), Paragraph("0.0634", cell_body), Paragraph("0.0299", cell_body), Paragraph("—", cell_body), Paragraph("1.25% (122)", cell_body)],
        [Paragraph("Random Baseline", cell_body), Paragraph("0.0000", cell_body), Paragraph("0.0000", cell_body), Paragraph("0.0000", cell_body), Paragraph("—", cell_body), Paragraph("46.19% (4,500)", cell_body)],
        [Paragraph("User-Based CF", cell_body), Paragraph("0.0091", cell_body), Paragraph("0.0909", cell_body), Paragraph("0.0459", cell_body), Paragraph("0.9227", cell_body), Paragraph("3.12% (304)", cell_body)],
        [Paragraph("Item-Based CF", cell_body), Paragraph("0.0066", cell_body), Paragraph("0.0661", cell_body), Paragraph("0.0411", cell_body), Paragraph("0.8922", cell_body), Paragraph("10.42% (1,015)", cell_body)],
        [Paragraph("SVD Matrix Fact.", cell_body), Paragraph("0.0039", cell_body), Paragraph("0.0386", cell_body), Paragraph("0.0209", cell_body), Paragraph("<b>0.8831</b>", cell_body), Paragraph("4.10% (399)", cell_body)],
        [Paragraph("Content-Based (TF-IDF)", cell_body), Paragraph("0.0017", cell_body), Paragraph("0.0165", cell_body), Paragraph("0.0081", cell_body), Paragraph("—", cell_body), Paragraph("<b>21.68% (2,112)</b>", cell_body)],
        [Paragraph("<b>⭐ Weighted Hybrid</b>", cell_body), Paragraph("<b>0.0096</b>", cell_body), Paragraph("<b>0.0964</b>", cell_body), Paragraph("<b>0.0521</b>", cell_body), Paragraph("—", cell_body), Paragraph("4.12% (401)", cell_body)],
        [Paragraph("NCF (PyTorch NeuMF)", cell_body), Paragraph("0.0074", cell_body), Paragraph("0.0744", cell_body), Paragraph("0.0377", cell_body), Paragraph("—", cell_body), Paragraph("8.49% (827)", cell_body)],
    ]
    t_res = Table(res_table, colWidths=[160, 80, 80, 90, 85, 115])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor("#E0F2FE")),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Key Takeaway:</b> The Weighted Hybrid achieved highest ranking accuracy across all metrics, while SVD achieved the lowest RMSE.", bullet_body))
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 10: LIVE DEMO
    story.extend(make_header("Deployment", "Interactive Streamlit Web Application & Live Demo"))
    demo_col1 = [
        Paragraph("<b>🍿 Existing User Recommendations:</b>", bullet_body),
        Paragraph("&bull; Select any user ID (1–610) &bull; View rating history & top genres.", bullet_body),
        Paragraph("&bull; Compare models side-by-side in real-time.", bullet_body),
        Paragraph("&bull; <b>👍/👎 Feedback Loop:</b> Reactions logged live to <code>feedback.csv</code>.", bullet_body),
    ]
    demo_col2 = [
        Paragraph("<b>❄️ Cold-Start Onboarding Engine:</b>", bullet_body),
        Paragraph("&bull; Instant fallback for zero-history users via Content TF-IDF.", bullet_body),
        Paragraph("&bull; 5 demo personas (Sci-Fi, Rom-Com, Crime, Animation, Drama).", bullet_body),
        Paragraph("&bull; Custom preference builder &bull; <40ms response time.", bullet_body),
    ]
    t_demo = Table([[demo_col1, demo_col2]], colWidths=[350, 350])
    t_demo.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_demo)
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Live Demo Link:</b> <code>localhost:8501</code> (or Streamlit Cloud) &bull; Backup video artifact: <code>docs/demo_video_storyboard.md</code>", bullet_body))
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 11: ETHICS & BIAS
    story.extend(make_header("Ethical Dimensions", "Popularity Bias Mitigation, Diversity & Privacy"))
    eth_col1 = [
        Paragraph("<b>Popularity Bias Mitigation:</b>", bullet_body),
        Paragraph("&bull; <b>The Danger:</b> Popularity baseline concentrates <b>56.77%</b> of slots into just 10 blockbusters, covering only 1.25% of catalog.", bullet_body),
        Paragraph("&bull; <b>Hybrid Solution:</b> Expands coverage to <b>401 unique movies (4.12%)</b>, tripling discovery while beating popularity accuracy.", bullet_body),
    ]
    eth_col2 = [
        Paragraph("<b>Filter Bubbles & Data Privacy:</b>", bullet_body),
        Paragraph("&bull; Blending collaborative and content signals prevents narrow genre echo chambers.", bullet_body),
        Paragraph("&bull; <b>Privacy by Design:</b> Zero PII in MovieLens; all feedback logged locally without external tracking.", bullet_body),
    ]
    t_eth = Table([[eth_col1, eth_col2]], colWidths=[350, 350])
    t_eth.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t_eth)
    story.append(PageBreak())

    # ---------------------------------------------------------------- SLIDE 12: CONCLUSION & QA
    story.append(Spacer(1, 20))
    story.append(Paragraph("CONCLUSION & TEAM ACKNOWLEDGMENTS", ParagraphStyle("Cat12", fontName="Helvetica-Bold", fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#2563EB"), spaceAfter=8)))
    story.append(Paragraph("Key Findings & Defense Q&A", ParagraphStyle("Title12", fontName="Helvetica-Bold", fontSize=24, alignment=TA_CENTER, textColor=colors.HexColor("#1E3A8A"), spaceAfter=14)))
    story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=16))

    concl_text = [
        Paragraph("&bull; <b>Hybrid Superiority:</b> Weighted Hybrid won on Recall@10 (0.0964) and NDCG@10 (0.0521), beating all baselines.", bullet_body),
        Paragraph("&bull; <b>Cold-Start Resilience:</b> Content TF-IDF provides instant fallback with 100% genre alignment.", bullet_body),
        Paragraph("&bull; <b>Production-Ready:</b> Full Streamlit application deployed with persistent feedback logging and benchmark dashboard.", bullet_body),
        Paragraph("&bull; <b>Open Source:</b> Reproducible code, splits, and models available on GitHub.", bullet_body),
    ]
    for c in concl_text:
        story.append(c)

    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Thank you! We warmly invite questions and discussion from the panel.</b>", ParagraphStyle("Thanks", fontName="Helvetica-Bold", fontSize=14, alignment=TA_CENTER, textColor=colors.HexColor("#1E3A8A"))))
    story.append(Paragraph("Reference defense guide: <code>docs/qa_document.md</code> (15 technical Q&As across all seats).", ParagraphStyle("QAGuide", fontName="Helvetica", fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceBefore=4)))

    doc.build(story, canvasmaker=SlideCanvas)
    print(f"Successfully generated 16:9 presentation slides PDF: {OUT_PDF}")


if __name__ == "__main__":
    build_slides()
