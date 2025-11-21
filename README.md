# Gluendo ADR Extractor

The **Gluendo ADR Extractor** is a Python-based system designed to automatically extract **Architecture Decision Record (ADR)** information from technical documents, including **PDF files, Word documents, images, and diagrams**.  
The project was developed during an internship at **Gluendo**, focusing on the automation of document analysis, diagram understanding, and structured knowledge extraction.

---

## 🚀 Features

### 📄 Document Processing
- Extracts text from PDF, Word, and image documents.
- Uses **EasyOCR** for multi-language OCR.
- Handles complex diagrams using object detection and arrow parsing.

### 📊 Diagram Understanding
- Detects:
  - Blocks / Nodes  
  - Arrows  
  - Relationships between objects  
- Classifies arrow types using **template matching**.
- Builds structured output describing object connections.

### 🖥️ PyQt5 Graphical Interface
- User-friendly desktop interface for:
  - Loading documents  
  - Viewing OCR results  
  - Inspecting detected arrows and blocks  
  - Exporting clean structured data  

### 📁 Output
Exports structured ADR information into:
- **CSV files**  
- **Readable text summaries**  
- Relation maps (object → target, arrow type, text)

---

## 🧠 Technologies Used

- **Python 3**
- **OpenCV** — for object detection, template matching, image preprocessing
- **EasyOCR** — for OCR text extraction
- **PyQt5** — for GUI
- **PaddleOCR / Detectron2 (optional modules)** — for enhanced diagram parsing
- **numpy, pandas** — data handling and CSV export

---

## 📂 Project Structure (example)

