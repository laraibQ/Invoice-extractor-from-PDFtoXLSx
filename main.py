import os
import logging
import tempfile

# Configure logging before importing extractors that call basicConfig.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# pdfminer DEBUG dumps every PDF operator and makes uploads crawl.
for noisy in (
    "pdfminer",
    "pdfminer.psparser",
    "pdfminer.pdfinterp",
    "pdfminer.pdfpage",
    "pdfminer.pdfdocument",
    "pdfminer.cmapdb",
    "pdfminer.converter",
    "pdfminer.pdfparser",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import pandas as pd

from text_invoice_parser import TextInvoiceParser

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "invoice-csv-studio-dev")

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "invoice_csv_studio_uploads")
OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), "invoice_csv_studio_outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

tesseract_path = os.environ.get("TESSERACT_PATH", None)
poppler_path = os.environ.get("POPPLER_PATH", None)

text_extractor = TextInvoiceParser(debug=False)

# Heavy OpenCV extractors are created only if the fast text pass fails.
_vision_extractor = None
_advanced_extractor = None
_fallback_extractor = None


def get_vision_extractor():
    global _vision_extractor
    if _vision_extractor is None:
        from vision_extractor import VisionInvoiceExtractor

        _vision_extractor = VisionInvoiceExtractor(
            tesseract_path=tesseract_path,
            debug=False,
        )
    return _vision_extractor


def get_advanced_extractor():
    global _advanced_extractor
    if _advanced_extractor is None:
        from invoice_table_extractor_advanced import AdvancedInvoiceExtractor

        _advanced_extractor = AdvancedInvoiceExtractor(
            tesseract_path=tesseract_path,
            debug=False,
        )
    return _advanced_extractor


def get_fallback_extractor():
    global _fallback_extractor
    if _fallback_extractor is None:
        from invoice_extractor import InvoiceTableExtractor

        _fallback_extractor = InvoiceTableExtractor(
            tesseract_path=tesseract_path,
            poppler_path=poppler_path,
            debug=False,
        )
    return _fallback_extractor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_usable_result(df: pd.DataFrame | None) -> bool:
    """Reject empty or obviously mangled extractions."""
    if df is None or df.empty:
        return False

    cols = [str(c).strip().lower() for c in df.columns]
    if any(c.startswith("column") for c in cols):
        return False
    if any(c in {"escription", "escriptio", "descriptio"} for c in cols):
        return False

    # New schema uses amount; older extractors may use Amount / total.
    amount_cols = [
        c
        for c in df.columns
        if str(c).lower() in {"amount", "total"} or "amount" in str(c).lower()
    ]
    if amount_cols:
        numeric = pd.to_numeric(df[amount_cols[0]], errors="coerce")
        if numeric.notna().sum() == 0:
            return False

    if len(df.columns) >= 7 and df.shape[0] >= 3:
        # Avoid flagging rich metadata tables; only check first content col.
        first_name = str(df.columns[0]).lower()
        if first_name in {"description", "item", "details"}:
            first_col = df.iloc[:, 0].astype(str)
            if (first_col.str.len() <= 3).mean() > 0.4:
                return False

    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))

    if not (file and allowed_file(file.filename)):
        flash("File type not allowed. Please upload a PDF file.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename or "uploaded_file.pdf")
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(pdf_path)

    export_format = (request.form.get("format") or "xlsx").strip().lower()
    if export_format not in {"xlsx", "csv"}:
        export_format = "xlsx"

    stem = os.path.splitext(filename)[0]
    output_name = f"{stem}.{export_format}"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)

    try:
        # Fast path first — text PDFs finish in well under a second.
        extractors = [
            ("text-layer", text_extractor),
            ("vision", None),
            ("advanced", None),
            ("fallback", None),
        ]

        results = None
        for name, extractor in extractors:
            if extractor is None:
                if name == "vision":
                    extractor = get_vision_extractor()
                elif name == "advanced":
                    extractor = get_advanced_extractor()
                else:
                    extractor = get_fallback_extractor()

            logger.info("Trying %s extraction for %s", name, filename)
            # Probe extractors with a temp csv path; final format is written below.
            probe_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{stem}.probe.csv")
            candidate = extractor.extract_from_pdf(pdf_path, probe_path)
            if is_usable_result(candidate):
                results = candidate
                logger.info("Accepted %s extraction (%s rows)", name, len(results))
                break
            logger.info("%s extraction rejected or empty", name)

        if not is_usable_result(results):
            flash("No valid table found in the invoice", "warning")
            return redirect(url_for("index"))

        writer = TextInvoiceParser()
        if export_format == "csv":
            writer._save_output(results, output_path)
            mimetype = "text/csv"
        else:
            writer._save_xlsx(results, output_path)
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        return send_file(
            output_path,
            mimetype=mimetype,
            download_name=output_name,
            as_attachment=True,
        )
    except Exception as e:
        logger.error("Error processing invoice: %s", e)
        flash(f"Error processing invoice: {e}", "error")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nInvoice CSV Studio is running.")
    print(f"Open: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
