"""Метрики хакатона на матрицах (дни x серии) + три фолда + проверка submission.
SMAPE по п.5.5 ТЗ (подтверждено организаторами): дни 0-против-0 дают ошибку 0 и ОСТАЮТСЯ в среднем.
В score.py из ТЗ это не так (mask = denominator > 0): там править."""
import numpy as np
import pandas as pd

PAIR = ["warehouse_name", "article"]
W = dict(smape=.50, wape=.20, rmse=.15, bias=.10, rmsle=.05)
HORIZON = 64

# (конец обучения, начало проверки, конец проверки); каждое окно = ровно 64 дня
FOLDS = {
    "A": ("2026-05-28", "2026-05-29", "2026-07-31"),  # самые свежие данные
    "B": ("2025-07-31", "2025-08-01", "2025-10-03"),  # тот же сезон, что и цель (2026-08-01..10-03)
    "C": ("2026-02-28", "2026-03-01", "2026-05-03"),  # другой период
}


def metrics(y, p):
    """y, p: массивы (дни, серии). Возвращает 5 метрик."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = (np.abs(y) + np.abs(p)) / 2
    err = np.where(d == 0, 0.0, np.abs(y - p) / np.where(d == 0, 1, d)) * 100
    ym, pm = y.mean(0), p.mean(0)
    m = ym != 0  # серии с mean(y)=0 исключаются из Bias
    return dict(
        smape=err.mean(0).mean(),
        wape=np.abs(y - p).sum() / np.abs(y).sum() * 100,
        rmse=float(np.sqrt(np.mean((y - p) ** 2))),
        bias=(np.abs(pm - ym)[m] / ym[m]).mean() * 100,
        rmsle=float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
    )


def score(m, base):
    """min-max: 0 = идеал, 1 = наивный бейзлайн, хуже бейзлайна клиппится на 1."""
    return sum(w * min(m[k] / base[k], 1) for k, w in W.items())


def to_matrix(df, col="quantity"):
    """Длинная таблица (date, склад, артикул, col) -> матрица (дни x серии), порядок серий = порядок появления."""
    pid = df.groupby(PAIR, sort=False).ngroup().to_numpy()
    tid = (df.date - df.date.min()).dt.days.to_numpy()
    M = np.zeros((tid.max() + 1, pid.max() + 1), np.float32)
    M[tid, pid] = df[col].to_numpy()
    pairs = df.drop_duplicates(PAIR)[PAIR].reset_index(drop=True)
    return M, pairs, pd.date_range(df.date.min(), periods=M.shape[0])


def check_submission(sub, pairs, first_day, horizon=HORIZON):
    """Проверка формата перед отправкой. pairs: таблица склад/артикул, first_day: первая дата прогноза."""
    assert list(sub.columns[:4]) == ["date", "warehouse_name", "article", "quantity"], "колонки/порядок"
    assert sub.quantity.notna().all() and (sub.quantity >= 0).all(), "NaN или отрицательные"
    dates = pd.to_datetime(sub.date).drop_duplicates().sort_values()
    assert len(dates) == horizon and (dates.diff().dropna() == pd.Timedelta(days=1)).all(), "даты"
    assert dates.iloc[0] == pd.Timestamp(first_day), "первая дата прогноза"
    assert len(sub) == len(pairs) * horizon and not sub.duplicated(["date"] + PAIR).any(), "число строк/дубли"
    assert len(sub[PAIR].drop_duplicates().merge(pairs)) == len(pairs), "набор пар"
    return True


def _test():
    y = np.array([[0, 2], [0, 4]], float)   # серия a: нули, серия b: 2 и 4
    p = np.array([[0, 2], [1, 2]], float)
    m = metrics(y, p)
    assert abs(m["smape"] - (100 + 100 / 3) / 2) < 1e-9   # a: (0+200)/2, b: (0+66.67)/2
    assert abs(m["wape"] - 50) < 1e-9
    assert abs(m["bias"] - 100 / 3) < 1e-9                # a исключена, b: |2-3|/3
    z = metrics(y, np.zeros_like(y))
    assert abs(z["smape"] - 100) < 1e-9                    # a: 0 против 0 = 0 (п.5.5), b: 200 -> (0+200)/2
    for a, b, c in FOLDS.values():
        assert (pd.Timestamp(c) - pd.Timestamp(b)).days + 1 == HORIZON
        assert pd.Timestamp(a) + pd.Timedelta(days=1) == pd.Timestamp(b)
    print("ok")


if __name__ == "__main__":
    _test()
