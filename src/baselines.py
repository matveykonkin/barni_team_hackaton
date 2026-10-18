"""Бейзлайны на трёх фолдах A/B/C + проверка идеи 'разреженный прогноз'.
Запуск из корня репозитория:  python -m src.baselines [train_full.parquet]
Результат: таблица в консоли и results/baselines.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import FOLDS, HORIZON, metrics, score, to_matrix


def predictions(Y, te, w0):
    """Прогнозы (64 x серии) по данным не позже индекса te (конец обучения). Y: (дни x серии)."""
    S = Y.shape[1]
    tile = lambda row: np.tile(row, (HORIZON, 1))
    h28, h56 = Y[te - 27:te + 1], Y[te - 55:te + 1]
    mean28, mean56 = h28.mean(0), h56.mean(0)
    share = (h56 > 0).mean(0)                                   # доля дней с продажей за 56 дней ~ P(продажа в день)
    pos = np.where(share > 0, h56.sum(0) / np.maximum((h56 > 0).sum(0), 1), 0)  # средний размер продажи
    out = {
        "naive_last": tile(Y[te]),
        "zero": np.zeros((HORIZON, S)),
        "mean28": tile(mean28),
        "mean56": tile(mean56),
    }
    for t in (0.3, 0.5, 0.7):
        out[f"sparse_mean28_t{t}"] = tile(np.where(share >= t, mean28, 0))
        out[f"sparse_pos_t{t}"] = tile(np.where(share >= t, pos, 0))
    # по дням недели: доля продаж в этот же день недели за последние 12 недель ~ P(продажа именно в этот день)
    ds = np.empty((HORIZON, S))
    for k in range(HORIZON):
        wd = (w0 + te + 1 + k) % 7
        ds[k] = (Y[[t for t in range(te - 83, te + 1) if (w0 + t) % 7 == wd]] > 0).mean(0)
    for t in (0.3, 0.4, 0.5):
        out[f"dow_pos_t{t}"] = np.where(ds >= t, pos, 0)
    return out


def main(path="data/processed/train_full.parquet"):
    df = pd.read_parquet(path, columns=["date", "warehouse_name", "article", "quantity"])
    Y, pairs, dates = to_matrix(df)
    idx = {d: i for i, d in enumerate(dates)}
    rows = []
    for fold, (tr_end, v0, v1) in FOLDS.items():
        te, a, b = idx[pd.Timestamp(tr_end)], idx[pd.Timestamp(v0)], idx[pd.Timestamp(v1)]
        truth = Y[a:b + 1]
        preds = predictions(Y, te, dates[0].weekday())
        # тот же период прошлого года (-364 дня = тот же день недели): среднее по окну, константой
        ly = Y[a - 364:b + 1 - 364].mean(0)
        preds["last_year_mean"] = np.tile(ly, (HORIZON, 1))
        base = metrics(truth, preds["naive_last"])
        for name, p in preds.items():
            m = metrics(truth, p)
            rows.append(dict(fold=fold, model=name, **m, score=score(m, base)))
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/baselines.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    for c in ("smape", "wape", "rmse", "bias", "rmsle", "score"):
        print(f"\n== {c} ==")
        print(res.pivot(index="model", columns="fold", values=c).assign(mean=lambda t: t.mean(axis=1)).sort_values("mean"))


if __name__ == "__main__":
    main(*sys.argv[1:])