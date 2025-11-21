# Gluendo ADR Extractor

The **Gluendo ADR Extractor** is a Python-based system designed to automatically extract **Architecture Decision Record (ADR)** information from technical documents, including **PDF files, Word documents, images, and diagrams**. The project was developed during an internship at **Gluendo**, focusing on automation of document analysis, diagram understanding, and structured knowledge extraction.

---

## 🚀 Features

### 📄 Document Processing
- Extracts text from PDF, Word (.docx), and image documents (PNG, JPG, TIFF).
- Uses **EasyOCR** for reliable multi-language OCR and falls back to Tesseract when configured.
- Performs PDF page conversion for scanned documents and extracts embedded text when present.

### 📊 Diagram Understanding
- Detects diagram components:
  - Blocks / Nodes
  - Arrows and connectors
  - Relationships and labels between objects
- Classifies arrow types using **template matching** and heuristics.
- Builds structured relation maps that capture (object → target, arrow type, associated text).

### 🖥️ PyQt5 Graphical Interface
- Desktop GUI for:
  - Loading and previewing documents
  - Inspecting OCR results and detected diagram objects
  - Correcting or validating extracted ADR fields
  - Exporting cleaned, structured outputs

### 📁 Output
Exports ADR information into:
- **CSV files** (tabular ADR records)
- **Readable text summaries** (human-friendly ADR snapshots)
- Relation maps and auxiliary JSON for downstream processing

---

## 🧰 Technologies Used

- **Python 3.8+**
- **OpenCV** — image processing, template matching, simple object detection
- **EasyOCR** — primary OCR engine for multi-language support
- **PyQt5** — graphical user interface
- **PaddleOCR / Detectron2 (optional)** — for advanced layout/diagram parsing
- **numpy, pandas** — data manipulation and CSV export

---

## ✅ What’s included

- Core extractor modules (OCR, diagram parser, relation mapper)
- A lightweight PyQt5 GUI for manual review and export
- Example input files and an examples/ folder demonstrating typical usage
- Configuration via `config.yaml` or environment variables

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/imovsesyan/Gluendo-ADR-Extractor.git
cd Gluendo-ADR-Extractor
```

2. Create a Python virtual environment and install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install system dependencies:
- Tesseract OCR (optional if using EasyOCR only, but recommended):
  - Ubuntu/Debian: `sudo apt install -y tesseract-ocr poppler-utils`
  - macOS (Homebrew): `brew install tesseract poppler`
- Poppler is required for `pdf2image` conversions when processing PDFs.

4. (Optional) Download model weights for the detector and place them in `models/`.
Configure the path in `config.yaml` or via environment variables (see Configuration).

---

## Configuration

The extractor reads settings from `config.yaml` (preferred) or environment variables.
Create a `config.yaml` in the project root with entries such as:

```yaml
TESSERACT_CMD: "/usr/bin/tesseract"
OCR_ENGINE: "easyocr"   # or "tesseract"
MODEL_PATH: "models/detector_weights.pth"
OCR_LANG: "en"          # EasyOCR language code(s), e.g. ["en","ru"]
OUTPUT_DIR: "results"
```

You can also set environment variables:

```bash
export TESSERACT_CMD="/usr/bin/tesseract"
export OCR_ENGINE="easyocr"
export MODEL_PATH="models/detector_weights.pth"
export OUTPUT_DIR="results"
```

---

## Usage

CLI examples:

Extract ADRs from a single PDF and write to CSV:

```bash
python run_extractor.py --input examples/sample-architecture.pdf --output results/adrs.csv
```

Process a folder of images and export detailed JSON and CSV:

```bash
python run_extractor.py --input examples/images/ --format json,csv --output results/
```

Start the GUI for manual review and export:

```bash
python gui/main.py
```

---

## Output format

The CSV output contains rows with fields similar to:

- id
- title
- context
- decision
- consequences
- authors
- date
- source_file
- page_number
- relation_map (path or JSON reference)

JSON exports include structured relation graphs and detection metadata.

---

## Troubleshooting

- OCR quality issues: increase DPI of inputs (300+), set OCR_ENGINE to tesseract or adjust EasyOCR language packs.
- Low text extraction from PDFs: for PDFs with embedded text prefer direct text extraction instead of OCR when available.
- Model loading errors: ensure MODEL_PATH is set and the ML framework (PyTorch / Paddle) is installed and matches the model format.
- GUI errors: verify PyQt5 is installed in the active environment.

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Fork the repository
2. Create a feature branch
3. Add or update tests and documentation
4. Open a pull request describing your changes

Please follow standard commit message practices and keep PRs focused.

---

## License

This repository currently has no license file. If you want to apply an open-source license, adding an MIT or Apache-2.0 LICENSE file is recommended.

---

## Acknowledgements

Built with open-source libraries and tools: EasyOCR, OpenCV, pdf2image, PyQt5, numpy, pandas, and optionally Detectron2 / PaddleOCR for advanced parsing.

---

## Contact 

Maintainer: imovsesyan
Repository: https://github.com/imovsesyan/Gluendo-ADR-Extractor
