from pathlib import Path
from src.features import extract_features, extract_spectral_band_features
from src.load import find_lvm_files, load_lvm
from src.config import COLUMNS, PROCESSED_DIR
import pandas as pd

SIGNAL_COLUMNS = [c for c in COLUMNS if c.startswith(("acc_", "vel_"))]
ACCELERATION_COLUMNS = [c for c in COLUMNS if c.startswith("acc_")]


def build_feature_table(files=None, include_spectral: bool = False) -> pd.DataFrame:
    """Uma linha por arquivo LVM; inclui features espectrais opcionalmente."""
    rows = []
    for file in files or find_lvm_files():
        df = load_lvm(file)
        row = {
            "file": file.name,
            "type": df["type"][0],
            "rotation_hz": df["rotation_hz"][0],
        }
        for column in SIGNAL_COLUMNS:
            features = extract_features(df[column].to_numpy())
            row.update({f"{column}_{name}": value for name, value in features.items()})
        if include_spectral:
            for column in ACCELERATION_COLUMNS:
                features = extract_spectral_band_features(df[column].to_numpy())
                row.update({f"{column}_{name}": value for name, value in features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def save_feature_table(
    output_path=None,
    include_spectral: bool = False,
) -> pd.DataFrame:
    """Gera e salva a tabela; a tabela espectral usa arquivo separado."""
    if output_path is None:
        filename = "features_with_spectral.csv" if include_spectral else "features.csv"
        output_path = PROCESSED_DIR / filename
    table = build_feature_table(include_spectral=include_spectral)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"{len(table)} linhas salvas em: {output_path}")
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera features temporais e espectrais dos LVMs.")
    parser.add_argument(
        "--with-spectral",
        action="store_true",
        help="Acrescenta RMS espectral em quatro bandas dos canais de aceleração.",
    )
    parser.add_argument("--output", type=Path, help="Caminho de saída opcional.")
    args = parser.parse_args()
    save_feature_table(output_path=args.output, include_spectral=args.with_spectral)
