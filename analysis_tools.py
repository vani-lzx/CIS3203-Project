from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def inspect_dataset(
    filename: str,
    target_column: str,
    feature_columns: list[str],
    separator: str = ";",
) -> dict:
    """Inspect a CSV dataset before training a machine learning model.

    Args:
        filename: CSV filename inside the project's data directory.
        target_column: Name of the column to predict.
        feature_columns: Names of the columns used as model inputs.
        separator: CSV delimiter, such as ";" or ",".

    Returns:
        Dataset information and validation results, or an error message.
    """
    if Path(filename).name != filename:
        return {"error": "Provide a filename inside the data directory."}

    path = DATA_DIR / filename
    if not path.is_file():
        return {"error": f"Dataset not found: {filename}"}

    if not feature_columns:
        return {"error": "Select at least one feature."}

    if target_column in feature_columns:
        return {"error": "The target must not be included in the features."}

    if len(feature_columns) != len(set(feature_columns)):
        return {"error": "Feature names must be unique."}

    try:
        df = pd.read_csv(path, sep=separator)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        return {"error": str(error)}

    selected_columns = [target_column] + feature_columns
    missing_columns = [
        column for column in selected_columns if column not in df.columns
    ]
    if missing_columns:
        return {
            "error": "Some requested columns do not exist.",
            "missing_columns": missing_columns,
            "available_columns": df.columns.tolist(),
        }

    return {
        "filename": filename,
        "rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "target_unique_values": int(df[target_column].nunique()),
        "missing_values": {
            column: int(df[column].isna().sum())
            for column in selected_columns
        },
        "column_types": {
            column: str(df[column].dtype)
            for column in selected_columns
        },
    }
