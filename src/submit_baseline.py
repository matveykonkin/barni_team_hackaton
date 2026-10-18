"""Черновой submission v0 по лучшему бейзлайну, обучение на всех данных.
Запуск:  python -m src.submit_baseline [train_full.parquet] [submission_v0.csv] [имя_бейзлайна]"""
import sys

import numpy as np
import pandas as pd

from src.baselines import predictions
from src.metrics import HORIZON, PAIR, check_submission, to_matrix


def main(path="data/processed/train_full.parquet", out="submission_v0.csv", name="sparse_mean28_t0.5"):
    Y, pairs, dates = to_matrix(pd.read_parquet(path, columns=["date", *PAIR, "quantity"]))
    p = predictions(Y, len(dates) - 1, dates[0].weekday())[name]          # (64, серии)
    days = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON)
    S = len(pairs)
    sub = pd.DataFrame({
        "date": np.repeat(days.strftime("%Y-%m-%d"), S),
        "warehouse_name": np.tile(pairs.warehouse_name.to_numpy(), HORIZON),
        "article": np.tile(pairs.article.to_numpy(), HORIZON),
        "quantity": p.ravel(),
    })
    check_submission(sub, pairs, days[0])
    sub.to_csv(out, index=False)
    print(out, len(sub), "строк;", days[0].date(), "->", days[-1].date(), "| доля нулей:", round((sub.quantity == 0).mean(), 3))


if __name__ == "__main__":
    main(*sys.argv[1:])