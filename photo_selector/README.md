# Photo Selector (Local-First AI Scaffolding)

A modular Python application for scanning local images and preparing an AI-assisted photo selection pipeline (wedding album use case).

## Features

- Fully local architecture (no cloud APIs)
- Cross-platform target (Windows/macOS/Linux)
- Scalable scaffolding for thousands of images
- SQLite-backed metadata and scores
- Streamlit UI for manual review and selection

## Project Structure

```text
photo_selector/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── ingestion/
│   │   └── scanner.py
│   │
│   ├── features/
│   │   ├── embedding.py
│   │   ├── aesthetic.py
│   │   ├── face_detection.py
│   │   ├── blur.py
│   │
│   ├── clustering/
│   │   └── cluster.py
│   │
│   ├── ranking/
│   │   └── ranker.py
│   │
│   ├── storage/
│   │   └── db.py
│   │
│   ├── ui/
│   │   └── streamlit_app.py
│   │
│   └── utils/
│       └── image_utils.py
│
├── data/
│   ├── embeddings/
│   ├── thumbnails/
│   └── database.sqlite
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd photo_selector
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app/main.py --input-folder /path/to/images
streamlit run app/ui/streamlit_app.py
```

## Notes

- ML modules are intentionally stubs for now (randomized placeholder outputs).
- `blur.py` uses OpenCV Laplacian variance for real blur scoring.
- Extend `features/` and `clustering/` with production models later.
