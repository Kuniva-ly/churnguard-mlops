"""Télécharge le dataset Telco Customer Churn depuis le repo officiel IBM.

Source primaire : repo IBM/telco-customer-churn-on-icp4d (GitHub).
Le fichier est public, libre d'usage à des fins éducatives et de recherche.

Le script vérifie l'intégrité du fichier par SHA-256 avant de l'écrire sur disque.
Si vous obtenez un mismatch, c'est probablement que le fichier upstream a été
modifié — auquel cas il faut mettre à jour la constante EXPECTED_SHA256 ou
récupérer le fichier d'une source alternative listée plus bas.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
EXPECTED_SIZE_BYTES = 970_457  # ~948 Ko
EXPECTED_ROWS = 7_043

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST = REPO_ROOT / "data" / "telco_churn.csv"

ALT_SOURCES = [
    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn (login requis)",
    "https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113",
]


def sha256_of(path: Path) -> str:
    """Retourne le SHA-256 hex d'un fichier."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        actual = sha256_of(DEST)
        if actual == EXPECTED_SHA256:
            print(f"[skip] fichier déjà présent et valide : {DEST}")
            return DEST
        print(f"[warn] checksum local différent ({actual[:12]}…), re-téléchargement")

    print(f"[..] téléchargement depuis {DATA_URL}")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"[err] échec du téléchargement : {exc}", file=sys.stderr)
        print("[hint] sources alternatives :", file=sys.stderr)
        for alt in ALT_SOURCES:
            print(f"       - {alt}", file=sys.stderr)
        sys.exit(1)

    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        print(
            f"[err] checksum invalide.\n"
            f"      attendu : {EXPECTED_SHA256}\n"
            f"      obtenu  : {actual_sha}",
            file=sys.stderr,
        )
        sys.exit(2)

    DEST.write_bytes(data)
    print(f"[ok] {len(data)} octets écrits dans {DEST}")
    print(f"[ok] checksum vérifié : {actual_sha}")

    # Sanity check rapide sur le nombre de lignes
    with DEST.open() as f:
        n_lines = sum(1 for _ in f)
    n_rows = n_lines - 1  # header
    if n_rows != EXPECTED_ROWS:
        print(
            f"[warn] nombre de lignes inattendu : {n_rows} (attendu {EXPECTED_ROWS})",
            file=sys.stderr,
        )
    else:
        print(f"[ok] {n_rows} lignes (hors header)")

    return DEST


if __name__ == "__main__":
    download()
