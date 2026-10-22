"""Funções para carregar sinais de vibração de arquivos .lvm."""

from pathlib import Path
import pandas as pd


def load_lvm(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Carrega um arquivo .lvm em um DataFrame.

    Nota: arquivos LVM podem variar conforme as opções de exportação.
    Ajuste `skiprows`, `sep` e outros argumentos conforme o cabeçalho
    e o delimitador do seu conjunto de dados.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Ponto de partida para arquivos tabulares delimitados por tabulação.
    # Alguns arquivos LVM têm cabeçalhos de metadados que exigem skiprows.
    return pd.read_csv(path, sep="\\t", engine="python", **read_csv_kwargs)
