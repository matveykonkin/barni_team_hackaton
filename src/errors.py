"""Где лучший бейзлайн теряет SMAPE: по складам, категориям, активности ряда, типу ошибки.
Запуск: python -m src.errors [train_full.parquet] [модель]   ->  results/errors_<фолд>.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import predictions
from src.metrics import FOLDS, PAIR, to_matrix


def smape_series(y, p):
    """SMAPE по каждой серии (0 против 0 = ошибка 0, как в п. 5.5 ТЗ)."""
    den = (np.abs(y) + np.abs(p)) / 2
    return (np.where(den == 0, 0, np.abs(y - p) / np.where(den == 0, 1, den)) * 100).mean(0)


def main(path="data/processed/train_full.parquet", name="sparse_mean28_t0.5"):
    df = pd.read_parquet(path, columns=["date", *PAIR, "quantity", "category_lvl_1"])
    Y, pairs, dates = to_matrix(df)
    cat = df.drop_duplicates(PAIR).set_index(PAIR).category_lvl_1.reindex(pd.MultiIndex.from_frame(pairs)).to_numpy()
    idx = {d: i for i, d in enumerate(dates)}
    Path("results").mkdir(exist_ok=True)
    for fold, (tr_end, v0, v1) in FOLDS.items():
        te, a, b = idx[pd.Timestamp(tr_end)], idx[pd.Timestamp(v0)], idx[pd.Timestamp(v1)]
        y, p = Y[a:b + 1], predictions(Y, te, dates[0].weekday())[name]
        share = (Y[te - 55:te + 1] > 0).mean(0)
        d = pairs.assign(
            category=cat,
            activity=pd.cut(share, [-1, 0, .05, .2, .5, 1.01], labels=["0", "0-5%", "5-20%", "20-50%", ">50%"]),
            smape=smape_series(y, p),
            missed=((y > 0) & (p == 0)).mean(0) * 100,        # % дней: продажа была, прогноз 0
            false_pos=((y == 0) & (p > 0)).mean(0) * 100,     # % дней: продаж нет, прогноз > 0
        )
        print(f"\n===== фолд {fold}, модель {name}, SMAPE по серии: среднее {d.smape.mean():.2f} =====")
        for col in ("activity", "warehouse_name", "category"):
            g = d.groupby(col, observed=True).agg(серий=("smape", "size"), smape=("smape", "mean"),
                                                  missed=("missed", "mean"), false_pos=("false_pos", "mean"))
            g["доля_потерь_%"] = g.серий * g.smape / d.smape.sum() * 100
            print(f"\n-- {col}\n", g.sort_values("доля_потерь_%", ascending=False).round(1).head(15))
        d.to_csv(f"results/errors_{fold}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
