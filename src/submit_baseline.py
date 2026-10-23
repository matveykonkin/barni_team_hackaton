"""Черновой недельный submission по бейзлайну: обучение на данных до LAST_TRAIN_DAY включительно.
Запуск:  python -m src.submit_baseline [train_full.parquet] [submission_v0.csv] [имя_бейзлайна]"""
import sys

import numpy as np
import pandas as pd

from src.baselines import predictions
from src.config import FIRST_WEEK, LAST_TRAIN_DAY, WEEKS
from src.metrics import check_submission, load, to_weeks


def main(path="train_full.parquet", out="submission_v0.csv", name="sparse_mean4_t0.5"):
    Y, pairs, idx = load(path)
    P = predictions(to_weeks(Y, 0, idx[pd.Timestamp(LAST_TRAIN_DAY)]))[name]   # дни после 26 июля в обучение НЕ берём
    weeks = pd.date_range(FIRST_WEEK, periods=WEEKS, freq="7D")
    sub = pd.DataFrame({
        "week": np.repeat(weeks.strftime("%Y-%m-%d"), len(pairs)),
        "warehouse_name": np.tile(pairs.warehouse_name.to_numpy(), WEEKS),
        "article": np.tile(pairs.article.to_numpy(), WEEKS),
        "quantity": P.ravel(),
    })
    check_submission(sub, pairs)
    sub.to_csv(out, index=False)
    print(out, len(sub), "строк;", sub.week.min(), "->", sub.week.max(), "| доля нулей:", round((sub.quantity == 0).mean(), 3))


if __name__ == "__main__":
    main(*sys.argv[1:])