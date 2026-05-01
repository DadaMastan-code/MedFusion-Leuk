"""
Script to download and organise leukemia datasets.
Run once before training: python notebooks/00_download_datasets.py

Datasets:
  1. ALL-IDB (Acute Lymphoblastic Leukemia Image Database)
  2. C-NMC 2019 (ISBI Challenge)
  3. Kaggle leukemia classification dataset
  4. PubMed abstracts for RAG indexing (via pymed)

Instructions per dataset are printed at runtime — some require
manual registration (C-NMC, TCIA) before download is possible.
"""

import os
import subprocess
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_LIT = Path("data/literature")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_LIT.mkdir(parents=True, exist_ok=True)


def download_kaggle_leukemia():
    print("\n[Kaggle] Downloading leukemia-classification dataset …")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "andrewmvd/leukemia-classification",
            "-p", str(DATA_RAW / "kaggle_leukemia"),
            "--unzip"
        ], check=True)
        print("[Kaggle] Done.")
    except FileNotFoundError:
        print("[Kaggle] kaggle CLI not found. Install: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"[Kaggle] Download failed: {e}")
        print("Ensure ~/.kaggle/kaggle.json is configured with your API key.")


def download_pubmed_abstracts(max_results: int = 500):
    """Fetch PubMed abstracts on leukemia for RAG indexing."""
    print("\n[PubMed] Fetching abstracts for RAG …")
    try:
        from pymed import PubMed
        pubmed = PubMed(tool="MedFusion-Leuk", email="researcher@example.com")
        queries = [
            "acute lymphoblastic leukemia diagnosis treatment",
            "acute myeloid leukemia clinical features prognosis",
            "chronic lymphocytic leukemia management",
            "chronic myeloid leukemia BCR-ABL tyrosine kinase",
            "leukemia blood smear morphology diagnosis",
        ]
        documents = []
        for query in queries:
            results = pubmed.query(query, max_results=max_results // len(queries))
            for article in results:
                abstract = article.abstract or ""
                if len(abstract.strip()) > 100:
                    documents.append({
                        "text": abstract,
                        "source": f"PubMed PMID:{article.pubmed_id} — {article.title or 'Untitled'}",
                    })

        import json
        out_path = DATA_LIT / "pubmed_abstracts.json"
        with open(out_path, "w") as f:
            json.dump(documents, f, indent=2)
        print(f"[PubMed] Saved {len(documents)} abstracts to {out_path}")

    except ImportError:
        print("[PubMed] pymed not installed. Run: pip install pymed")


def build_faiss_index():
    """Index downloaded abstracts into FAISS for RAG retrieval."""
    import json
    from modules.rag.retriever import MedicalRetriever

    abs_path = DATA_LIT / "pubmed_abstracts.json"
    if not abs_path.exists():
        print("[FAISS] No abstracts found. Run download_pubmed_abstracts() first.")
        return

    with open(abs_path) as f:
        documents = json.load(f)

    print(f"[FAISS] Indexing {len(documents)} documents …")
    retriever = MedicalRetriever()
    retriever.build_from_texts(documents)
    retriever.save()
    print("[FAISS] Index saved.")


if __name__ == "__main__":
    print("=== MedFusion-Leuk Dataset Setup ===")
    print("\nNote: C-NMC 2019 and ALL-IDB datasets require manual registration.")
    print("Download links:")
    print("  ALL-IDB  : http://homes.di.unimi.it/scotti/all/")
    print("  C-NMC    : https://wiki.cancerimagingarchive.net/")
    print("  LISC     : http://lisc-dataset.ir/")

    download_kaggle_leukemia()
    download_pubmed_abstracts()
    build_faiss_index()
    print("\n=== Setup complete ===")
