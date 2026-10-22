import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://zenodo.org/records/14388093/files"

CONDITIONS = [
    "Baseline",
    "Bearing",
    "Bearing+Unbalance",
    "Misalignment_NDE",
    "Unbalance_DE",
    "Unbalance_NDE",
]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def download(fault_condition):
    filename = f"{fault_condition}.zip"
    zip_path = RAW / filename
    output_dir = RAW / fault_condition

    if (output_dir / ".extracted").exists():
        print(f"{fault_condition} já foi extraído.")
        return

    if not zip_path.exists():
        url = f"{URL}/{filename}?download=1"

        with requests.get(
            url, stream=True, timeout=(30, 120)
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as file:
                for chunk in tqdm(
                    response.iter_content(1024 * 1024),
                    total=(total // (1024 * 1024) + 1)
                    if total else None,
                    desc=filename,
                    unit="MB",
                ):
                    if chunk:
                        file.write(chunk)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    (output_dir / ".extracted").write_text(
        "OK", encoding="utf-8"
    )
    print(f"{fault_condition} concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        download(condition)