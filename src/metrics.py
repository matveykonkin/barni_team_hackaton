"""Метрики хакатона на матрицах (недели x серии), недельная агрегация и проверка submission.
SMAPE по п.5.5 ТЗ: точки 0-против-0 дают ошибку 0 и ОСТАЮТСЯ в среднем (совпадает с score.py из ТЗ)."""
import numpy as np
import pandas as pd

from src.config import FIRST_WEEK, FOLDS, PAIR, W, WEEKS


def metrics(y, p):
    """y, p: массивы (недели, серии). Возвращает 5 метрик."""
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
    """min-max: 0 = идеал, 1 = наивный бейзлайн, хуже бейзлайна клиппится на 1. Финал у организаторов ранговый."""
    return sum(w * min(m[k] / base[k], 1) for k, w in W.items())


def to_matrix(df, col="quantity"):
    """Длинная таблица (date, склад, артикул, col) -> дневная матрица (дни x серии), порядок серий = порядок появления."""
    pid = df.groupby(PAIR, sort=False).ngroup().to_numpy()
    tid = (df.date - df.date.min()).dt.days.to_numpy()
    M = np.zeros((tid.max() + 1, pid.max() + 1), np.float32)
    M[tid, pid] = df[col].to_numpy()
    pairs = df.drop_duplicates(PAIR)[PAIR].reset_index(drop=True)
    dates = pd.date_range(df.date.min(), periods=M.shape[0])
    assert dates[0].weekday() == 0, "день 0 должен быть понедельником, иначе недели сдвинутся"
    return M, pairs, dates


def to_weeks(Y, a, b):
    """Недельные суммы дней [a, b] (включительно) дневной матрицы. День 0 = понедельник, поэтому a % 7 == 0."""
    assert a % 7 == 0 and (b - a + 1) % 7 == 0, "окно должно состоять из целых недель пн-вс"
    return Y[a:b + 1].reshape(-1, 7, Y.shape[1]).sum(1)


def load(path="train_full.parquet"):
    """parquet -> (дневная матрица, пары, {дата: индекс дня})."""
    Y, pairs, dates = to_matrix(pd.read_parquet(path, columns=["date", *PAIR, "quantity"]))
    return Y, pairs, {d: i for i, d in enumerate(dates)}


def check_submission(sub, pairs):
    """Формат из п.4 ТЗ. pairs: таблица склад/артикул (эталонный набор)."""
    assert list(sub.columns) == ["week", "warehouse_name", "article", "quantity"], "колонки/порядок"
    assert sub.quantity.notna().all() and (sub.quantity >= 0).all(), "NaN или отрицательные"
    weeks = pd.to_datetime(sub.week).drop_duplicates().sort_values()
    assert len(weeks) == WEEKS and (weeks.dt.day_of_week == 0).all(), "недели: 9 понедельников"    assert (weeks.diff().dropna() == pd.Timedelta(days=7)).all() and weeks.iloc[0] == pd.Timestamp(FIRST_WEEK), "шаг 7 дней, первая неделя"
    assert len(sub) == len(pairs) * WEEKS and not sub.duplicated(["week"] + PAIR).any(), "число строк/дубли"
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
        a, b, c = map(pd.Timestamp, (a, b, c))
        assert a.weekday() == 6 and b.weekday() == 0 and c.weekday() == 6, "фолд: вс / пн / вс"
        assert (c - b).days + 1 == 7 * WEEKS and a + pd.Timedelta(days=1) == b
    Y = np.arange(28, dtype=float).reshape(14, 2)           # 2 недели x 2 серии
    assert (to_weeks(Y, 0, 13) == [[Y[:7, 0].sum(), Y[:7, 1].sum()], [Y[7:, 0].sum(), Y[7:, 1].sum()]]).all()
    pairs = pd.DataFrame({"warehouse_name": ["w1", "w1"], "article": ["a1", "a2"]})
    weeks = pd.date_range(FIRST_WEEK, periods=WEEKS, freq="7D").strftime("%Y-%m-%d")
    ok = pd.DataFrame({"week": np.repeat(weeks, 2), "warehouse_name": "w1", "article": ["a1", "a2"] * WEEKS, "quantity": 1.0})
    assert check_submission(ok, pairs)
    for bad in (ok.assign(quantity=-1.0), ok.iloc[:-1], ok.assign(week=ok.week.replace(FIRST_WEEK, "2026-07-28")), ok.assign(article="a1")):
        try:
            check_submission(bad, pairs)
        except AssertionError:
            continue
        raise SystemExit("check_submission пропустил плохой файл")
    print("ok")


if __name__ == "__main__":
    _test()