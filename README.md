# Invoice CSV Studio

Extract line-item tables from invoice PDFs and download them as **Excel (.xlsx)** or **CSV**.

## Features

- Fast text-layer parsing for digital PDFs (no Poppler required for that path)
- Fallback extractors for harder / scanned layouts (OCR + OpenCV when installed)
- Web UI with drag-and-drop upload and format choice (XLSX or CSV)
- CLI scripts for batch or debugging use

## Requirements

- Python 3.11+
- Packages in `requirements.txt`
- Optional for scanned PDFs:
  - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  - [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (for `pdf2image`)

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `TESSERACT_PATH` | Full path to `tesseract` / `tesseract.exe` |
| `POPPLER_PATH` | Folder with Poppler binaries |
| `SESSION_SECRET` | Flask session secret (set this in production) |
| `PORT` | Web port (default `5000`) |

## Setup

```bash
cd invoice-csv-studio
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

System packages (only needed for OCR / image conversion):

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

## Run the web app

```bash
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), upload a PDF, choose **Excel** or **CSV**, download the result.

## Command-line tools

```bash
# Fast text-layer parser (recommended for digital PDFs)
python -c "from text_invoice_parser import TextInvoiceParser; TextInvoiceParser().extract_from_pdf('samples/Invoice_INV-2026-049.pdf', 'out.xlsx')"

# General extractor
python invoice_extractor.py path/to/invoice.pdf --output out.csv

# Advanced heuristics
python invoice_table_extractor_advanced.py path/to/invoice.pdf -o out.csv -d

# Vision / layout extractor
python vision_extractor.py path/to/invoice.pdf -o out.csv -d
```

## Project layout

| Path | Role |
|------|------|
| `main.py` | Flask web app |
| `text_invoice_parser.py` | Primary text-layer extractor + XLSX/CSV writer |
| `vision_extractor.py` | Computer-vision layout pass |
| `invoice_table_extractor_advanced.py` | Advanced heuristics |
| `invoice_extractor.py` | Multi-method fallback |
| `table_detector.py` | Image table helpers |
| `table_extraction_enhancement.py` | Cleanup / scoring helpers |
| `templates/` / `static/` | Web UI |
| `samples/` | Example invoice PDF |

## Notes

- Digital (text) PDFs usually extract quickly through the text-layer path.
- Scanned PDFs need Tesseract (and often Poppler).
- Use `-d` / `--debug` on CLI scripts when a file fails.

## License

MIT
