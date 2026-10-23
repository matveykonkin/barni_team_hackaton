"""Недельные бейзлайны на трёх фолдах A/B/C + проверка идеи 'разреженный прогноз'.
Запуск из корня репозитория:  python -m src.baselines [train_full.parquet]
Результат: таблицы в консоли и results/baselines.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FOLDS, WEEKS
from src.metrics import load, metrics, score, to_weeks


def predictions(Wk):
    """Прогнозы (9 x серии) по недельной истории Wk (недели x серии), последняя строка = последняя известная неделя."""
    assert len(Wk) >= 52, "для last_year нужно не меньше 52 недель истории"
    tile = lambda row: np.tile(row, (WEEKS, 1))
    h4, h13 = Wk[-4:], Wk[-13:]
    mean4, share = h4.mean(0), (h13 > 0).mean(0)    # доля недель с продажей за 13 недель ~ P(продажа в неделе)
    pos = np.where(share > 0, h13.sum(0) / np.maximum((h13 > 0).sum(0), 1), 0)   # средняя ненулевая неделя
    out = {"naive_last": tile(Wk[-1]), "zero": np.zeros((WEEKS, Wk.shape[1])), "mean4": tile(mean4),
           "mean13": tile(h13.mean(0)), "last_year": Wk[-52:-52 + WEEKS]}   # те же 9 недель на 52 недели раньше
    for t in (0.3, 0.5, 0.7):
        out[f"sparse_mean4_t{t}"] = tile(np.where(share >= t, mean4, 0))
        out[f"sparse_pos_t{t}"] = tile(np.where(share >= t, pos, 0))
    return out


def main(path="train_full.parquet"):
    Y, _, idx = load(path)
    rows = []
    for fold, (tr_end, v0, v1) in FOLDS.items():
        te = idx[pd.Timestamp(tr_end)]
        assert idx[pd.Timestamp(v0)] == te + 1
        truth = to_weeks(Y, te + 1, idx[pd.Timestamp(v1)])
        assert len(truth) == WEEKS
        preds = predictions(to_weeks(Y, 0, te))
        base = metrics(truth, preds["naive_last"])
        print(f"фолд {fold}: доля нулевых недель в факте {(truth == 0).mean():.3f}")
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