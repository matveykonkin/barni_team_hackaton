import numpy as np
import pandas as pd

from src.metrics import PAIR

ROLL_WINDOWS = [7, 14, 28, 56] # среднее за последние n дней
LAGS = [1, 2, 3, 7, 14, 28] # продажи n дней назад (чтобы например уловить сезонность)

FEATURE_COLS = [
    # лаги
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
    # роллинги
    # - средние за n дней
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
    # - стандартное отклонение за последние n дней
    "roll_std_7", "roll_std_14", "roll_std_28", "roll_std_56",
    # - сколько дней из n дней продажи были нулевыми
    "roll_zero_ratio_7", "roll_zero_ratio_14", "roll_zero_ratio_28", "roll_zero_ratio_56",
    # - среднее количество продаж только по тем дням, когда реально продавалось
    "pos_56",
    # - доля дней из последних 56, когда продавались
    "share_56",
    # - количество дней с последней продажи
    "days_since_last_sale",
    # - остатки на складе n дней назад
    "stock_lag_1", "stock_lag_2", "stock_lag_3", "stock_lag_7", "stock_lag_14", "stock_lag_28",
    # - средние остатки за полседние n дней
    "stock_roll_mean_7", "stock_roll_mean_14",
    # - доля дней без записи в стоке за 180 дней
    "stock_missing_mean_180",
    # - доля дней с остатком 0 за 180 дней
    "stock_zero_mean_180",
    # - stock в последний день
    "stock_last",
    # - данные по календарю (день недели, месяц, день месяца, выходной или нет)
    "dow", "month", "day_of_month", "is_weekend",
    # категориальные признаки
    "warehouse_name", "category_lvl_1", "category_lvl_2", "cities_name"
]

# категориальные признаки
CATEGORICAL = ["warehouse_name", "category_lvl_1", "category_lvl_2", "cities_name"]


def _pos_56(s):
    nz = s[s > 0]
    return nz.tail(56).mean() if len(nz) else 0.0 # tail - берет последние 56 строк


def _dls(q):
    idx = np.arange(len(q))
    last_sale_idx = np.where(q>0, idx, -1)
    last_sale_idx = np.maximum.accumulate(last_sale_idx) # проходится по каждому и если видит индекс больше предыдущего, то пишет индекс этот
    result = np.full(len(q), 999, dtype=int) # массив из 999
    mask = last_sale_idx >= 0 # смотрим, где индекс больше 0, то есть быда продажа
    result[mask] = idx[mask] - last_sale_idx[mask] # вычитаем только там, где была продажа
    return result

def build_features(df, cutoff_date):

    hist = df[df.date <= cutoff_date].copy()
    hist = hist.sort_values(PAIR + ["date"]).reset_index(drop=True)
    g = hist.groupby(PAIR, sort=False)

    # словарь с признаками
    feats = {}

    for lag in LAGS:
        feats[f"lag_{lag}"] = g["quantity"].shift(lag) # shift - берет данные из предыдущей строки

    for roll in ROLL_WINDOWS:
        feats[f"roll_mean_{roll}"] = g["quantity"].transform(
            lambda s, roll = roll: s.rolling(roll, min_periods = 1).mean() # rolling - создает окно из последних roll значений quantity
        )
        feats[f"roll_std_{roll}"] = g["quantity"].transform(
            lambda s, roll = roll: s.rolling(roll, min_periods = 1).std()
        )
        feats[f"roll_zero_ratio_{roll}"] = g["quantity"].transform(
           lambda s, roll=roll: (s == 0).rolling(roll, min_periods = 1).mean()
        )

    feats["pos_56"] = g["quantity"].transform(_pos_56)
    feats["share_56"] = g["quantity"].transform(
        lambda s: (s>0).tail(56).mean()
    )

    feats["days_since_last_sale"] = g["quantity"].transform(_dls)

    for lag in LAGS:
        feats[f"stock_lag_{lag}"] = g["stock"].shift(lag)

    for roll in [7, 14]:
        feats[f"stock_roll_mean_{roll}"] = g["stock"].transform(
            lambda s, roll=roll: s.rolling(roll, min_periods=1).mean()
        )

    feats["stock_missing_mean_180"] = g["stock_missing"].transform(lambda s: s.tail(180).mean())
    feats["stock_zero_mean_180"] = g["stock_zero"].transform(lambda s: s.tail(180).mean())
    feats["stock_last"] = hist["stock"].values

    feats["warehouse_name"] = hist["warehouse_name"].values
    feats["category_lvl_1"] = hist["category_lvl_1"].values
    feats["category_lvl_2"] = hist["category_lvl_2"].values
    feats["cities_name"] = hist["cities_name"].values

    # признаки в таблицу из словаря
    out = pd.DataFrame(feats)
    out[PAIR] = hist[PAIR].values
    out["date"] = hist["date"].values

    # Заполняем инфу об днях (как признаки тоже)
    out["dow"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["is_weekend"] = (out["date"].dt.weekday >= 5).astype(int)

    # Заполнение NaN (только лаги/роллинги)
    fill_zero = [c for c in out.columns if c.startswith(("lag_", "roll_"))]
    out[fill_zero] = out[fill_zero].fillna(0)

    return out # (N, F): N - все дни х все серии, F - все признаки


def build_training_set(df, cutoff_date, h, max_days = None):
    X_all = build_features(df, cutoff_date)

    target = df[PAIR + ["date", "quantity"]].rename(columns={"quantity": "y"})
    target["date"] = target["date"] - pd.Timedelta(days=h) # мы сдвигаем даты назад, чтобы получить y для T+h на дату, которая в X_all
    merged = X_all.merge(target, on=PAIR + ["date"], how="inner") # ключ - серия + дата, inner - склеиваем если ключ есть в обоих таблицах
    merged = merged.dropna(subset=["y"]) # удаляет строки, где y = NaN
    if max_days:
        max_date = merged["date"].max() - pd.Timedelta(days=max_days)
        merged = merged[merged["date"] >= max_date]

    X = merged[FEATURE_COLS].reset_index(drop=True)
    for c in CATEGORICAL:
        X[c] = X[c].astype("category").cat.codes.astype("int32")

    y = np.log1p(merged["y"].to_numpy()) # log1p = log(1 + y), чтобы модель училась равномерно
    return X, y
