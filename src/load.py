from pathlib import Path
import re
import pandas as pd
from config import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, COLUMNS

def find_lvm_files():
    """Encontra todos os arquivos .lvm dentro de data/raw."""
    return sorted(RAW_DIR.rglob("*.lvm"))


def load_lvm(file_path):
    """Lê um arquivo LVM e retorna os sinais numéricos."""
    file_path = Path(file_path)

    with file_path.open(
        "r", encoding="utf-8-sig", errors="replace"
    ) as file:
        lines = file.readlines()

    header_ends = [
        i for i, line in enumerate(lines)
        if "***End_of_Header***" in line
    ]

    if len(header_ends) < 2:
        raise ValueError(
            f"Não encontrei os dois cabeçalhos em {file_path}"
        )

    # Pula a linha de nomes das colunas.
    data_start = header_ends[1] + 2

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=data_start,
        header=None,
        usecols=range(11),
        names=COLUMNS,
    )

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Tipo de falha (ou baseline) = pasta filha de data/raw que contém o arquivo.
    folders = file_path.relative_to(RAW_DIR).parts[:-1]
    df["type"] = folders[0]

    # Rotação (Hz) = pasta com nome no formato "<número>Hz", ex.: "20Hz".
    df["rotation_hz"] = next(
        int(m.group(1))
        for part in folders
        if (m := re.fullmatch(r"(\d+)Hz", part))
    )

    return df


if __name__ == "__main__":
    files = find_lvm_files()
    print(f"Arquivos encontrados: {len(files)}")

    if files:
        df = load_lvm(files[0])

        print("\nArquivo:", files[0].name)
        print(df.head())
        print("\nDimensões:", df.shape)
        print("\nValores ausentes:")
        print(df.isna().sum())

