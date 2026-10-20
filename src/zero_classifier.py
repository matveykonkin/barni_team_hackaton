"""Zero classifier: p = ожидаемая доля дней с продажей в горизонте 64 дня -> гейт для прогноза величины.
Почему доля, а не "будет ли хоть одна продажа": SMAPE усредняется по дням серии, константа c на серии
с долей q положительных дней даёт ~ (1-q)*200 + q*e, ноль даёт q*200 -> обнулять выгодно при малом q.
Обучение: внутренние cutoff T-64-28k (метки целиком <= T), tau подбирается на самом свежем внутреннем окне,
затем модель переобучается на всех внутренних cutoff и применяется в точке T. Validation fold не трогается.
Запуск из корня репозитория:  python -m src.zero_classifier [train_full.parquet]
Результат: results/zero_classifier.csv, results/zero_segments.csv, oof/zero_{A,B,C}.parquet"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import FOLDS, HORIZON, PAIR, metrics, score, to_matrix

WIN = 180                    # максимум истории для признаков (CLAUDE.md)
STEP, N_CUT = 28, 8          # внутренние cutoff: T - 64 - 28k, k = 0..7
TAUS = np.round(np.arange(0.1, 0.95, 0.05), 2)
SEG_TAUS = np.r_[0.0, TAUS, 1.01]   # для сегмента можно "оставить всё" / "обнулить всё"
SEG_EDGES = [0, 0.05, 0.2, 0.5, 0.8]
SEG_NAMES = ["0", "0-5", "5-20", "20-50", "50-80", "80-100"]   # по share56, как в baselines.py
LGB = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42, deterministic=True, verbose=-1)
ROUNDS = 300
LGB_DAY = {**LGB, "objective": "binary", "min_data_in_leaf": 500}


def segment(share56):
    return np.searchsorted(SEG_EDGES, share56, side="left")   # 0 -> "0", (0, .05] -> "0-5", ...


def tsb(H, a, b=0.1):
    """TSB по окну H (дни x серии): p затухает на каждом нулевом дне, z обновляется только в дни продаж."""
    pos = H > 0
    p = pos.mean(0)
    z = H.sum(0) / np.maximum(pos.sum(0), 1)
    for h, s in zip(H, pos):
        p = np.where(s, p + a * (1 - p), (1 - a) * p)
        z = np.where(s, z + b * (h - z), z)
    return p, z


def features(Y, S, te, start, wh, art):
    """Признаки серий в точке te (индекс дня, включительно) только по окну [te-179, te]. start: дата te+1."""
    H, pos = Y[te - WIN + 1:te + 1], Y[te - WIN + 1:te + 1] > 0
    f = {}
    for w in (7, 14, 28, 56, 91, 180):
        f[f"share{w}"], f[f"mean{w}"] = pos[-w:].mean(0), H[-w:].mean(0)
    n = pos.sum(0)
    f["pos56"] = np.where(pos[-56:].any(0), H[-56:].sum(0) / np.maximum(pos[-56:].sum(0), 1), 0)  # как в baselines
    f["recency"] = np.where(pos.any(0), np.argmax(pos[::-1], 0), WIN)        # = длина текущей полосы нулей
    f["pos_streak"] = np.where((~pos).any(0), np.argmax(~pos[::-1], 0), WIN)
    f["age"] = np.where(pos.any(0), WIN - np.argmax(pos, 0), 0)               # дней с первой продажи в окне
    Hp = np.where(pos, H, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = dict(pos_mean=np.nanmean(Hp, 0), pos_median=np.nanmedian(Hp, 0), pos_std=np.nanstd(Hp, 0),
                     pos_max=np.nanmax(Hp, 0), pos_q90=np.nanquantile(Hp, .9, 0))
    f.update({k: np.nan_to_num(v) for k, v in stats.items()})
    f["adi"] = WIN / np.maximum(n, 1)
    f["cv2"] = np.where(f["pos_mean"] > 0, (f["pos_std"] / np.maximum(f["pos_mean"], 1e-9)) ** 2, 0)
    for a in (0.05, 0.1, 0.2):
        f[f"tsb{int(a * 100)}"] = tsb(H, a)[0]
    st, sm, sz = (S[c][te - WIN + 1:te + 1] for c in ("stock", "stock_missing", "stock_zero"))
    f["stock"], f["stock_zero"] = st[-1], sz[-1]
    f["stock_zero_streak"] = np.where((sz == 0).any(0), np.argmax(sz[::-1] == 0, 0), WIN)
    f["stock_missing180"], f["stock_zero28"] = sm.mean(0), sz[-28:].mean(0)
    avail = sz[-56:] == 0
    f["share56_instock"] = (pos[-56:] & avail).sum(0) / np.maximum(avail.sum(0), 1)
    # кросс-серийные агрегаты на той же точке te
    for name, key in (("art", art), ("wh", wh)):
        f[f"{name}_share56"] = (np.bincount(key, f["share56"]) / np.bincount(key))[key]
    f["month"], f["doy"] = np.full(len(wh), start.month), np.full(len(wh), start.dayofyear)
    f["wh"] = wh
    return pd.DataFrame(f)


def day_features(Y, te, start, F):
    """Строки (день горизонта h x серия s) только для серий с продажами за 180 дней; + признаки дня недели."""
    H = Y[te - WIN + 1:te + 1]
    wd = (start.weekday() - WIN + np.arange(WIN)) % 7          # день недели каждого дня окна
    by_dow = lambda w, fn: np.stack([fn(H[-w:][wd[-w:] == d]) for d in range(7)])
    sh12, sh25 = by_dow(84, lambda x: (x > 0).mean(0)), by_dow(175, lambda x: (x > 0).mean(0))
    mn25 = by_dow(175, lambda x: x.mean(0))
    live = np.flatnonzero(F.share180.to_numpy() > 0)
    h, s = np.repeat(np.arange(HORIZON), len(live)), np.tile(live, HORIZON)
    dow = (start.weekday() + h) % 7
    X = F.iloc[s].reset_index(drop=True).astype(np.float32)
    X["h"], X["dow"], X["dom"] = h, dow, (start + pd.to_timedelta(h, "D")).day
    X["dow_share12"], X["dow_share25"], X["dow_mean25"] = sh12[dow, s], sh25[dow, s], mn25[dow, s]
    return X, h, s


def fit_predict(X, q, Xte):
    """p из разных источников. Метка q в [0, 1]: LGB cross_entropy, LR через пару строк с весами q / 1-q."""
    out = {"share56": Xte.share56.to_numpy(), "tsb10": Xte.tsb10.to_numpy()}
    cols = [c for c in X if c != "wh"]
    tr = lambda D: np.log1p(D[cols].clip(lower=0))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    A = tr(X)
    lr.fit(pd.concat([A, A]), np.r_[np.ones(len(q)), np.zeros(len(q))],
           logisticregression__sample_weight=np.r_[q, 1 - q])
    out["lr"] = lr.predict_proba(tr(Xte))[:, 1]
    m = lgb.train(LGB, lgb.Dataset(X, q, categorical_feature=["wh"]), ROUNDS)
    out["lgb"] = m.predict(Xte)
    return out, m


def run(truth, pred, base):
    m = metrics(truth, np.broadcast_to(pred, truth.shape))
    return m, score(m, base)


def pick_tau(truth, p, mag, seg, base):
    """Глобальный tau по score, затем покоординатно tau для каждого сегмента (старт = глобальный)."""
    sc = lambda t: run(truth, np.where(p >= t, mag, 0), base)[1]
    g = min(TAUS, key=sc)
    taus = np.full(len(SEG_NAMES), g)
    for s in range(len(SEG_NAMES)):
        def sc_s(t, s=s):
            tt = taus.copy()
            tt[s] = t
            return sc(tt[seg])
        best = min(SEG_TAUS, key=sc_s)
        if sc_s(best) < sc_s(taus[s]):   # меняем только при строгом улучшении
            taus[s] = best
    return g, taus


def main(path="data/processed/train_full.parquet"):
    df = pd.read_parquet(path, columns=["date", *PAIR, "quantity", "stock", "stock_missing", "stock_zero"])
    for c in ("stock", "stock_missing", "stock_zero"):
        df[c] = df[c].fillna(0).astype(np.float32)
    Y, pairs, dates = to_matrix(df)
    S = {c: to_matrix(df, c)[0] for c in ("stock", "stock_missing", "stock_zero")}
    wh, art = pairs.warehouse_name.factorize()[0], pairs.article.factorize()[0]
    idx = {d: i for i, d in enumerate(dates)}
    feats = lambda te: features(Y, S, te, dates[te + 1], wh, art)
    target = lambda te: (Y[te + 1:te + 1 + HORIZON] > 0).mean(0)
    win = lambda te: Y[te + 1:te + 1 + HORIZON]

    def day_model(train_cuts, te, Fte):
        """Target 2: P(продажа в день h) -> матрица (64 x серии); серии без продаж за 180 дней -> 0."""
        parts = [day_features(Y, c, dates[c + 1], F[c]) for c in train_cuts]
        y = np.concatenate([Y[c + 1 + h, s] > 0 for c, (_, h, s) in zip(train_cuts, parts)]).astype(float)
        m = lgb.train(LGB_DAY, lgb.Dataset(pd.concat([x for x, _, _ in parts]), y, categorical_feature=["wh"]), ROUNDS)
        X, h, s = day_features(Y, te, dates[te + 1], Fte)
        P = np.zeros((HORIZON, len(wh)))
        P[h, s] = m.predict(X)
        return P

    rows, segrows = [], []
    Path("results").mkdir(exist_ok=True)
    Path("oof").mkdir(exist_ok=True)
    for fold, (tr_end, v0, v1) in FOLDS.items():
        T = idx[pd.Timestamp(tr_end)]
        cuts = [c for c in (T - HORIZON - STEP * k for k in range(N_CUT)) if c >= WIN - 1]
        F = {c: feats(c) for c in cuts}
        inner, rest = cuts[0], cuts[1:]
        p_in, _ = fit_predict(pd.concat([F[c] for c in rest]), np.concatenate([target(c) for c in rest]), F[inner])
        Fv = feats(T)
        p_v, model = fit_predict(pd.concat(F.values()), np.concatenate([target(c) for c in cuts]), Fv)
        p_in["lgb_day"], p_v["lgb_day"] = day_model(rest, inner, F[inner]), day_model(cuts, T, Fv)
        t_in, t_v = win(inner), win(T)
        assert t_v.shape[0] == HORIZON and dates[T + 1] == pd.Timestamp(v0) and dates[T + HORIZON] == pd.Timestamp(v1)
        b_in, b_v = (metrics(t, np.broadcast_to(Y[c], t.shape)) for t, c in ((t_in, inner), (t_v, T)))
        s_in, s_v = segment(F[inner].share56.to_numpy()), segment(Fv.share56.to_numpy())
        for src, p in p_v.items():
            brier = float(np.mean((np.broadcast_to(p, t_v.shape) - (t_v > 0)) ** 2))   # против дневного факта
            for mag in ("mean28", "pos56", "pos_median"):
                M = Fv[mag].to_numpy()
                g, taus = pick_tau(t_in, p_in[src], F[inner][mag].to_numpy(), s_in, b_in)
                oracle = min(TAUS, key=lambda t: run(t_v, np.where(p >= t, M, 0), b_v)[1])
                cands = {
                    ("fixed", "0.5"): np.where(p >= 0.5, M, 0),
                    ("global", str(g)): np.where(p >= g, M, 0),
                    ("segment", "/".join(map(str, taus))): np.where(p >= taus[s_v], M, 0),
                    ("ORACLE_global", str(oracle)): np.where(p >= oracle, M, 0),   # не честный OOF, только потолок
                }
                if mag == "pos56":
                    cands[("p_weighted", "-")] = p * M
                for (strat, tau), pred in cands.items():
                    pred = np.broadcast_to(pred, t_v.shape)
                    m, s = run(t_v, pred, b_v)
                    rows.append(dict(fold=fold, p_src=src, mag=mag, strategy=strat, tau=tau, **m, score=s, brier=brier))
                    for k, name in enumerate(SEG_NAMES):
                        col = s_v == k
                        if col.any():
                            segrows.append(dict(fold=fold, p_src=src, mag=mag, strategy=strat, segment=name,
                                                n_series=int(col.sum()), demand_share=t_v[:, col].sum() / t_v.sum(),
                                                **metrics(t_v[:, col], pred[:, col])))
        days = pd.date_range(v0, periods=HORIZON)
        pd.DataFrame({"date": np.repeat(days, len(pairs)),
                      "warehouse_name": np.tile(pairs.warehouse_name.to_numpy(), HORIZON),
                      "article": np.tile(pairs.article.to_numpy(), HORIZON),
                      "p": np.tile(p_v["lgb"], HORIZON),
                      "p_day": p_v["lgb_day"].ravel()}).to_parquet(f"oof/zero_{fold}.parquet", index=False)
        imp = pd.Series(model.feature_importance("gain"), model.feature_name()).sort_values(ascending=False)
        print(f"fold {fold}: cuts={len(cuts)}, top features:", ", ".join(imp.index[:8]))
    res = pd.DataFrame(rows)
    res.to_csv("results/zero_classifier.csv", index=False)
    pd.DataFrame(segrows).to_csv("results/zero_segments.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.3f}".format, "display.max_rows", 200)
    for c in ("score", "smape"):
        print(f"\n== {c} ==")
        print(res.pivot_table(index=["p_src", "mag", "strategy"], columns="fold", values=c)
              .assign(mean=lambda t: t.mean(axis=1)).sort_values("mean"))


if __name__ == "__main__":
    main(*sys.argv[1:])
