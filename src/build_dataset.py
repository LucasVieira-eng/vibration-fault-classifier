from pathlib import Path
from src.features import extract_features
from src.load import COLUMNS, find_lvm_files, load_lvm
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SIGNAL_COLUMNS = [c for c in COLUMNS if c.startswith(("acc_", "vel_"))]


def build_feature_table(files=None) -> pd.DataFrame:
    """Uma linha por arquivo .lvm: features de cada canal + rótulos."""
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
        rows.append(row)
    return pd.DataFrame(rows)


def save_feature_table(output_path=PROCESSED_DIR / "features.csv") -> pd.DataFrame:
    """Gera a tabela de features e salva em CSV."""
    table = build_feature_table()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"{len(table)} linhas salvas em: {output_path}")
    return table


if __name__ == "__main__":
    save_feature_table()