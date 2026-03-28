from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.config import INDEX_JSON_PATH


def _load_rows() -> list[dict]:
    if not INDEX_JSON_PATH.exists():
        return []
    try:
        payload = json.loads(INDEX_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {INDEX_JSON_PATH}: {exc}")
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        return [item for item in payload["images"] if isinstance(item, dict)]
    return []


def main() -> None:
    st.set_page_config(page_title="Photo Selector", layout="wide")
    st.title("Local AI Photo Selector")
    st.caption("Scaffold UI for reviewing and selecting photos")

    selected_folder = st.text_input("Image folder path", value="")
    if selected_folder:
        folder_path = Path(selected_folder)
        if folder_path.exists() and folder_path.is_dir():
            st.success(f"Folder selected: {folder_path}")
        else:
            st.warning("Folder not found. Pipeline requires a valid local path.")

    rows = _load_rows()
    if not rows:
        st.info("No images found in index JSON. Run the pipeline first with app/main.py.")
        return

    st.subheader(f"Images ({len(rows)})")
    columns = st.columns(3)
    for idx, row in enumerate(rows):
        col = columns[idx % 3]
        with col:
            image_path = row.get("path", "")
            st.image(image_path, use_container_width=True)
            st.write(f"Score: {row.get('final_score')}")
            st.write(f"Aesthetic: {row.get('aesthetic_score')}")
            st.write(f"Blur: {row.get('blur_score')}")
            st.write(f"Faces: {row.get('face_count')}")
            st.checkbox("Select", key=f"select_{row.get('id')}")


if __name__ == "__main__":
    main()
