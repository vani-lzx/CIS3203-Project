from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "student-por.csv"


def summarize_dataset() -> dict:
    """Summarize the student dataset and performance differences by school.

    Returns:
        Dataset dimensions, missing values, and performance statistics
        for each school. A final grade of at least 10 counts as passing.
    """
    df = pd.read_csv(DATA_PATH, sep=";")
    schools = {}

    for school, group in df.groupby("school"):
        passing = int((group["G3"] >= 10).sum())
        failing = int((group["G3"] < 10).sum())

        schools[str(school)] = {
            "students": int(len(group)),
            "average_final_grade": float(group["G3"].mean()),
            "passing_students": passing,
            "failing_students": failing,
            "pass_rate": float(passing / len(group)),
        }

    return {
        "dataset": "UCI Student Performance - Portuguese",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "target_definition": "Failing = G3 < 10; passing = G3 >= 10",
        "schools": schools,
    }
