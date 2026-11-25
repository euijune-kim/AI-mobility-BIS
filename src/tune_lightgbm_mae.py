"""Tune LightGBM for MAE on the competition training data.

The requested shuffled 5-fold CV is reported as a diagnostic. Parameter
selection uses date-held-out folds because the competition data are temporal.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from sklearn.model_selection import KFold


CAT_FEATURES = ["route_id", "from_stop", "to_stop"]
NUM_FEATURES = [
    "route_type",
    "direction",
    "from_lat",
    "from_lon",
    "to_lat",
    "to_lon",
    "dlat",
    "dlon",
    "stop_seq",
    "dist_m",
    "arrive_hour",
    "is_weekend",
]
BASE_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 5,
    "max_depth": -1,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_estimators": 1800,
    "random_state": 0,
    "verbosity": -1,
    "n_jobs": 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/final/train.parquet"))
    parser.add_argument("--workbook", type=Path, default=Path("Attempts/6th_attempt.xlsx"))
    parser.add_argument("--template", type=Path, default=Path("Attempts/4th_attempt.xlsx"))
    parser.add_argument("--trials", type=int, default=30)
    return parser.parse_args()


def add_features(frame):
    out = frame.copy()
    day = pd.to_datetime(out["date"]).dt.day_of_week
    out["is_weekend"] = (day >= 5).astype("int8")
    out["dlat"] = out["to_lat"] - out["from_lat"]
    out["dlon"] = out["to_lon"] - out["from_lon"]
    return out


def prepare(frame, categories, features):
    out = add_features(frame)[features].copy()
    for column in CAT_FEATURES:
        if column in out:
            out[column] = pd.Categorical(out[column], categories=categories[column])
    return out


def make_model(params, n_estimators=None):
    model_params = {**BASE_PARAMS, **params}
    if n_estimators is not None:
        model_params["n_estimators"] = n_estimators
    return lgb.LGBMRegressor(**model_params)


def fit_and_predict(train_x, train_y, valid_x, valid_y, params):
    model = make_model(params)
    model.fit(
        train_x,
        train_y,
        eval_X=valid_x,
        eval_y=valid_y,
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict(valid_x, num_iteration=model.best_iteration_)
    score = float(np.mean(np.abs(valid_y.to_numpy() - pred)))
    return score, model.best_iteration_


def shuffled_cv(data, categories, features, params):
    x = prepare(data, categories, features)
    y = data["tt_sec"].reset_index(drop=True)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    importances = pd.Series(0.0, index=features)
    iterations = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        model = make_model(params)
        model.fit(
            x.iloc[train_idx],
            y.iloc[train_idx],
            eval_X=x.iloc[valid_idx],
            eval_y=y.iloc[valid_idx],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict(x.iloc[valid_idx], num_iteration=model.best_iteration_)
        score = float(np.mean(np.abs(y.iloc[valid_idx].to_numpy() - pred)))
        scores.append(score)
        iterations.append(model.best_iteration_)
        importances += pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=features,
        )
        print(f"5-Fold {fold}/5 MAE={score:.4f}, trees={model.best_iteration_}", flush=True)
    importances /= len(scores)
    return scores, importances, iterations


def date_cv(data, categories, features, params):
    rows = []
    for date in sorted(data["date"].unique()):
        train = data[data["date"] != date]
        valid = data[data["date"] == date]
        train_x = prepare(train, categories, features)
        valid_x = prepare(valid, categories, features)
        score, iteration = fit_and_predict(
            train_x, train["tt_sec"], valid_x, valid["tt_sec"], params
        )
        day_name = "토요일" if pd.Timestamp(date).day_of_week == 5 else "평일"
        rows.append(
            {
                "date": date,
                "day": day_name,
                "MAE": score,
                "best_iteration": iteration,
            }
        )
    weekday_mae = float(np.mean([row["MAE"] for row in rows if row["day"] == "평일"]))
    saturday_mae = float(np.mean([row["MAE"] for row in rows if row["day"] == "토요일"]))
    weighted_mae = (3 * weekday_mae + saturday_mae) / 4
    return rows, weekday_mae, saturday_mae, weighted_mae


def saturday_cv(data, categories, features, params):
    saturday_dates = [
        date
        for date in sorted(data["date"].unique())
        if pd.Timestamp(date).day_of_week == 5
    ]
    if len(saturday_dates) != 1:
        raise ValueError(f"Expected one Saturday training date, found {len(saturday_dates)}.")
    valid_date = saturday_dates[0]
    train = data[data["date"] != valid_date]
    valid = data[data["date"] == valid_date]
    return fit_and_predict(
        prepare(train, categories, features),
        train["tt_sec"],
        prepare(valid, categories, features),
        valid["tt_sec"],
        params,
    )


def select_features(importances):
    feature_names = list(importances.index)
    remove_count = max(1, math.ceil(len(feature_names) * 0.10))
    bottom = set(importances.sort_values(kind="stable").index[:remove_count])
    remove = {
        name
        for name, gain in importances.items()
        if gain <= 0 or name in bottom
    }
    return [name for name in feature_names if name not in remove], sorted(remove)


def export_workbook(args, data, baseline_cv, importance, removed, baseline_date, best_date, study):
    if args.workbook.exists():
        raise FileExistsError(f"Refusing to overwrite {args.workbook}")
    workbook = load_workbook(args.template)
    records = workbook["실험 기록"]
    row_id = "C06-01"
    record = [
        row_id,
        "LightGBM Optuna 튜닝",
        "5-Fold gain 하위 10% 제거 후 Optuna; 전체 날짜 CV로 최종 선택",
        "LightGBM",
        best_date["weekday_mae"],
        best_date["saturday_mae"],
        best_date["weighted_mae"],
        (
            "기본 모델 유지: Optuna 후보보다 4일 CV 우수/동률"
            if best_date["name"].startswith("Baseline")
            else "선택: 전체 날짜 CV 4일 가중 MAE 최저"
        ),
        json.dumps(best_date["params"], ensure_ascii=False, sort_keys=True),
        None,
        None,
        None,
        (
            f"기본 5-fold 평균 MAE={np.mean(baseline_cv):.4f}; "
            f"하위 제거 피처={', '.join(removed)}; "
            f"Optuna {len(study.trials)}회; 최종 선택={best_date['name']}. "
            "test 정답이 없어 test MAE는 산출하지 않음."
        ),
        datetime.now(),
    ]
    records.append(record)
    records.auto_filter.ref = f"A1:N{records.max_row}"

    cv_sheet = workbook.create_sheet("6차 검증")
    cv_sheet.append(["구분", "fold/date", "요일", "MAE (초)", "best_iteration"])
    for fold, score in enumerate(baseline_cv, start=1):
        cv_sheet.append(["기본 5-Fold KFold", fold, "무작위", score, None])
    cv_sheet.append(["기본 날짜 CV", "평일 평균", "월·화·수", baseline_date["weekday_mae"], None])
    cv_sheet.append(["기본 날짜 CV", "토요일", "토요일", baseline_date["saturday_mae"], None])
    cv_sheet.append(["기본 날짜 CV", "4일 가중 평균", "3평일+1토요일", baseline_date["weighted_mae"], None])
    cv_sheet.append(["최적 날짜 CV", "평일 평균", "월·화·수", best_date["weekday_mae"], None])
    cv_sheet.append(["최적 날짜 CV", "토요일", "토요일", best_date["saturday_mae"], None])
    cv_sheet.append(["최적 날짜 CV", "4일 가중 평균", "3평일+1토요일", best_date["weighted_mae"], None])
    for row in best_date["folds"]:
        cv_sheet.append(["최적 날짜 CV", row["date"], row["day"], row["MAE"], row["best_iteration"]])
    for cell in cv_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    cv_sheet.freeze_panes = "A2"
    cv_sheet.auto_filter.ref = f"A1:E{cv_sheet.max_row}"
    for column, width in {"A": 24, "B": 22, "C": 20, "D": 16, "E": 20}.items():
        cv_sheet.column_dimensions[column].width = width

    importance_sheet = workbook.create_sheet("Gain 중요도")
    importance_sheet.append(["피처", "5-fold 평균 gain", "선택"])
    for feature, gain in importance.sort_values(ascending=False).items():
        importance_sheet.append([feature, float(gain), "제거" if feature in removed else "사용"])
    for cell in importance_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    importance_sheet.freeze_panes = "A2"
    importance_sheet.auto_filter.ref = f"A1:C{importance_sheet.max_row}"
    importance_sheet.column_dimensions["A"].width = 24
    importance_sheet.column_dimensions["B"].width = 22
    importance_sheet.column_dimensions["C"].width = 14

    tuning_sheet = workbook.create_sheet("Optuna 30 trials")
    tuning_sheet.append(["trial", "4일 환산 MAE (평일 고정)", "평일 MAE", "토요일 MAE", "파라미터"])
    for trial in study.trials:
        if trial.value is not None:
            weekday, saturday = trial.user_attrs["date_scores"]
            tuning_sheet.append(
                [
                    trial.number,
                    trial.value,
                    weekday,
                    saturday,
                    json.dumps(trial.params, ensure_ascii=False, sort_keys=True),
                ]
            )
    for cell in tuning_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    tuning_sheet.freeze_panes = "A2"
    tuning_sheet.auto_filter.ref = f"A1:E{tuning_sheet.max_row}"
    for column, width in {"A": 12, "B": 24, "C": 16, "D": 16, "E": 90}.items():
        tuning_sheet.column_dimensions[column].width = width

    candidate_sheet = workbook.create_sheet("최종 후보 4일 CV")
    candidate_sheet.append(["후보", "trial", "평일 MAE", "토요일 MAE", "4일 가중 MAE", "파라미터"])
    for candidate in best_date["candidate_results"]:
        candidate_sheet.append(
            [
                candidate["name"],
                candidate.get("trial_number"),
                candidate["weekday_mae"],
                candidate["saturday_mae"],
                candidate["weighted_mae"],
                json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
            ]
        )
    for cell in candidate_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    candidate_sheet.freeze_panes = "A2"
    candidate_sheet.auto_filter.ref = f"A1:F{candidate_sheet.max_row}"
    for column, width in {"A": 22, "B": 12, "C": 16, "D": 16, "E": 20, "F": 90}.items():
        candidate_sheet.column_dimensions[column].width = width

    summary = workbook["탐색 요약"]
    summary.append(["6차 기본 5-Fold CV MAE", f"{np.mean(baseline_cv):.4f}초"])
    summary.append(["6차 기본 날짜 CV", f"평일 {baseline_date['weekday_mae']:.4f} / 토요일 {baseline_date['saturday_mae']:.4f} / 가중 {baseline_date['weighted_mae']:.4f}초"])
    summary.append(["6차 최적 날짜 CV", f"평일 {best_date['weekday_mae']:.4f} / 토요일 {best_date['saturday_mae']:.4f} / 가중 {best_date['weighted_mae']:.4f}초"])
    summary.append(["6차 날짜 CV 가중 MAE 변화", f"{best_date['weighted_mae'] - baseline_date['weighted_mae']:+.4f}초"])
    summary.append(["6차 최종 선택 모델", best_date["name"]])
    summary.append(["6차 Optuna 적용 피처", ", ".join(feature for feature in importance.index if feature not in removed)])
    summary.append(["6차 gain 제거 후보", ", ".join(removed) if removed else "없음"])
    summary.append(["6차 최종 선택 파라미터", json.dumps(best_date["params"], ensure_ascii=False, sort_keys=True)])
    summary.append(["6차 Optuna 목적", "토요일 홀드아웃 MAE 최소화; 상위 3개를 전체 날짜 CV로 재평가"])
    summary.append(["6차 검증 참고", "무작위 5-Fold는 같은 날짜의 행이 train/validation에 섞일 수 있어 참고 지표이며, 날짜 CV로 최종 선택"])
    workbook.save(args.workbook)


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = pd.read_parquet(args.train)
    if "tt_sec" not in data or data["date"].nunique() < 2:
        raise ValueError("Training data must include tt_sec and multiple dates.")

    features = CAT_FEATURES + NUM_FEATURES
    augmented = add_features(data)
    categories = {
        feature: pd.Index(augmented[feature].dropna().unique())
        for feature in CAT_FEATURES
    }

    print(f"Train rows={len(data):,}; dates={sorted(data['date'].unique())}")
    baseline_cv, importance, baseline_iterations = shuffled_cv(
        data, categories, features, BASE_PARAMS
    )
    baseline_mae = float(np.mean(baseline_cv))
    print(f"기본 5-Fold 평균 MAE: {baseline_mae:.4f}")
    print("기본 모델 gain 중요도:")
    print(importance.sort_values(ascending=False).to_string(float_format=lambda value: f"{value:.2f}"))

    selected_features, removed = select_features(importance)
    print(f"제거 피처: {removed}")
    print(f"선택 피처 수: {len(selected_features)} / {len(features)}")

    baseline_params = {
        key: BASE_PARAMS[key]
        for key in [
            "num_leaves",
            "min_child_samples",
            "max_depth",
            "colsample_bytree",
            "subsample",
            "reg_alpha",
            "reg_lambda",
        ]
    }
    baseline_folds, base_weekday, base_saturday, base_weighted = date_cv(
        data, categories, features, baseline_params
    )
    baseline_date = {
        "weekday_mae": base_weekday,
        "saturday_mae": base_saturday,
        "weighted_mae": base_weighted,
    }
    print(
        "기본 날짜 CV: "
        f"평일={base_weekday:.4f}, 토요일={base_saturday:.4f}, "
        f"4일 가중 MAE={base_weighted:.4f}"
    )

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        saturday_mae, iteration = saturday_cv(
            data, categories, selected_features, params
        )
        weekday_mae = baseline_date["weekday_mae"]
        weighted_mae = (3 * weekday_mae + saturday_mae) / 4
        trial.set_user_attr("date_scores", [weekday_mae, saturday_mae])
        trial.set_user_attr("best_iteration", iteration)
        return weighted_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def report_trial(study, trial):
        weekday_mae, saturday_mae = trial.user_attrs["date_scores"]
        print(
            f"Optuna {trial.number + 1}/{args.trials}: "
            f"4일 MAE={trial.value:.4f}, 평일={weekday_mae:.4f}, "
            f"토요일={saturday_mae:.4f}, best={study.best_value:.4f}",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=False,
        callbacks=[report_trial],
    )
    candidates = [
        {
            "name": f"Optuna trial {trial.number}",
            "trial_number": trial.number,
            "params": trial.params,
        }
        for trial in sorted(
            (trial for trial in study.trials if trial.value is not None),
            key=lambda item: item.value,
        )[:3]
    ]
    candidates.append(
        {
            "name": "Baseline (all features)",
            "params": baseline_params,
            "folds": baseline_folds,
            "weekday_mae": base_weekday,
            "saturday_mae": base_saturday,
            "weighted_mae": base_weighted,
        }
    )
    candidate_results = []
    for candidate in candidates:
        if "folds" not in candidate:
            folds, weekday_mae, saturday_mae, weighted_mae = date_cv(
                data, categories, selected_features, candidate["params"]
            )
            candidate.update(
                {
                    "folds": folds,
                    "weekday_mae": weekday_mae,
                    "saturday_mae": saturday_mae,
                    "weighted_mae": weighted_mae,
                }
            )
        candidate_results.append(candidate)
        print(
            f"{candidate['name']} full date CV: "
            f"평일={candidate['weekday_mae']:.4f}, "
            f"토요일={candidate['saturday_mae']:.4f}, "
            f"4일 MAE={candidate['weighted_mae']:.4f}",
            flush=True,
        )
    best_date = min(candidate_results, key=lambda candidate: candidate["weighted_mae"])
    best_date["candidate_results"] = candidate_results

    print("\n튜닝 전후 검증 요약")
    print(f"5-Fold 기준 MAE: {baseline_mae:.4f}초")
    print(
        f"날짜 CV 4일 가중 MAE: {base_weighted:.4f} -> "
        f"{best_date['weighted_mae']:.4f} "
        f"({best_date['weighted_mae'] - base_weighted:+.4f}초)"
    )
    print(f"평일 MAE: {base_weekday:.4f} -> {best_date['weekday_mae']:.4f}")
    print(f"토요일 MAE: {base_saturday:.4f} -> {best_date['saturday_mae']:.4f}")
    print(f"최종 선택: {best_date['name']}")
    print("최적 파라미터:", json.dumps(best_date["params"], ensure_ascii=False, sort_keys=True))

    export_workbook(
        args,
        data,
        baseline_cv,
        importance,
        removed,
        baseline_date,
        best_date,
        study,
    )
    print(f"결과 워크북: {args.workbook}")


if __name__ == "__main__":
    main()
