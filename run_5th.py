"""
5th Attempt: LightGBM MAE Optimization with 5-Fold K-Fold, Feature Selection, and Optuna Tuning
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 경고 메시지 숨김
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def find_train_dataset():
    """데이터셋 경로를 우선순위에 따라 탐색합니다."""
    candidates = [
        'data/final/train.parquet',
        'data/final/train.csv',
        '실습용/train.parquet',
        '실습용/train.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("학습 데이터셋(train.parquet / train.csv)을 찾을 수 없습니다.")


def load_and_preprocess_data():
    """데이터셋을 불러오고 기본 파생 피처를 생성합니다."""
    train_path = find_train_dataset()
    print(f"[1/5] 데이터셋 로드: {train_path}")
    
    if train_path.endswith('.parquet'):
        df = pd.read_parquet(train_path)
    else:
        df = pd.read_csv(train_path)
        
    print(f"      원본 데이터 크기: {df.shape}")

    # 기본 파생 변수 생성
    PEAK_HOURS = {7, 8, 9, 17, 18, 19}
    dt = pd.to_datetime(df['date'])
    df['dow'] = dt.dt.day_of_week
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_peak'] = df['arrive_hour'].isin(PEAK_HOURS).astype(int)
    df['interval'] = np.select(
        [df['dow'] == 5, df['dow'] == 6],
        [df['interval_sat'], df['interval_sun']],
        default=df['interval_wd']
    )
    df['dlat'] = df['to_lat'] - df['from_lat']
    df['dlon'] = df['to_lon'] - df['from_lon']

    target_col = 'tt_sec'
    drop_cols = ['row_id', 'date', target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 범주형 컬럼 명시
    cat_cols = ['route_id', 'route_nm', 'vehicle_id', 'from_stop', 'from_name', 'to_stop', 'to_name']
    cat_cols = [c for c in cat_cols if c in feature_cols]

    X = df[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    y = df[target_col].copy()

    print(f"      전처리 완료: 피처 수 {len(feature_cols)}개 (범주형: {len(cat_cols)}개, 수치형: {len(feature_cols) - len(cat_cols)}개)")
    return df, X, y, feature_cols, cat_cols


def run_baseline_5fold(X, y, feature_cols, cat_cols):
    """기본 파라미터로 5-Fold 교차 검증(K-Fold)을 수행하고 Gain 피처 중요도를 계산합니다."""
    print("\n[2/5] 기본 파라미터 5-Fold K-Fold 교차 검증 및 피처 중요도 분석 시작...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_maes = []
    feature_importances = np.zeros(len(feature_cols))

    base_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'importance_type': 'gain',
        'learning_rate': 0.1,
        'n_estimators': 300,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**base_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        f_mae = mean_absolute_error(y_val, val_pred)
        fold_maes.append(f_mae)
        feature_importances += model.feature_importances_ / 5.0
        print(f"      Fold {fold} MAE: {f_mae:.4f} (Best Iteration: {model.best_iteration_})")

    baseline_cv_mae = float(np.mean(fold_maes))
    baseline_oof_mae = float(mean_absolute_error(y, oof_preds))
    elapsed = time.time() - start_time

    print(f"\n   >>> [기준 MAE] 5-Fold 평균 MAE: {baseline_cv_mae:.4f} 초 (OOF MAE: {baseline_oof_mae:.4f} 초, 소요 시간: {elapsed:.1f}초)")

    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'gain': feature_importances
    }).sort_values('gain', ascending=False).reset_index(drop=True)

    print("\n   >>> [피처 중요도 (Gain 기준 전체 순위)]")
    print(fi_df.to_string(index=True))

    return baseline_cv_mae, fold_maes, fi_df, kf


def select_features(fi_df):
    """Gain 기준 중요도가 하위 10%이거나 0인 피처를 제거합니다."""
    print("\n[3/5] 피처 선택 (Gain 하위 10% 또는 Gain <= 0 피처 제거)...")
    
    total_feats = len(fi_df)
    gain_threshold_10pct = np.percentile(fi_df['gain'], 10)
    
    # 제거 조건: gain이 0 이하이거나, gain이 하위 10% 분위수 이하
    to_drop_mask = (fi_df['gain'] <= 0) | (fi_df['gain'] <= gain_threshold_10pct)
    dropped_df = fi_df[to_drop_mask].copy()
    kept_df = fi_df[~to_drop_mask].copy()

    dropped_features = dropped_df['feature'].tolist()
    selected_features = kept_df['feature'].tolist()

    print(f"      전체 피처 수: {total_feats}개")
    print(f"      하위 10% 분위수 임계값: {gain_threshold_10pct:.4f}")
    print(f"      제거된 피처 수: {len(dropped_features)}개")
    print(f"      제거된 피처 목록: {dropped_features}")
    print(f"      유지된 피처 수: {len(selected_features)}개")
    print(f"      유지된 피처 목록: {selected_features}")

    return selected_features, dropped_features, kept_df, dropped_df, gain_threshold_10pct


def tune_with_optuna(X_selected, y, kf, n_trials=30):
    """Optuna를 사용하여 7개 하이퍼파라미터를 30회 튜닝합니다."""
    print(f"\n[4/5] Optuna 하이퍼파라미터 튜닝 시작 (총 {n_trials}회 탐색)...")
    print("      튜닝 대상 파라미터: num_leaves, min_child_samples, max_depth,")
    print("                        colsample_bytree, subsample, reg_alpha, reg_lambda")

    trial_records = []

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'learning_rate': 0.1,
            'n_estimators': 300,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'subsample_freq': 1,
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        maes = []
        for train_idx, val_idx in kf.split(X_selected, y):
            X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X_selected.iloc[val_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
            )
            val_pred = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, val_pred))

        mean_mae = float(np.mean(maes))
        
        # 기록용
        rec = {'trial_number': trial.number, 'cv_mae': mean_mae}
        rec.update(trial.params)
        trial_records.append(rec)

        print(f"      [Trial {trial.number + 1:2d}/{n_trials}] CV MAE: {mean_mae:.4f} | "
              f"depth={params['max_depth']}, leaves={params['num_leaves']}, "
              f"colsample={params['colsample_bytree']:.2f}, subsample={params['subsample']:.2f}")

        return mean_mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    optuna_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    optuna_elapsed = time.time() - optuna_start

    print(f"\n   >>> Optuna 튜닝 완료! (총 소요 시간: {optuna_elapsed:.1f}초)")
    print(f"   >>> 최적 CV MAE: {study.best_value:.4f} 초")
    print(f"   >>> 최적 파라미터:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"       - {k}: {v:.6f}" if v < 0.01 else f"       - {k}: {v:.4f}")
        else:
            print(f"       - {k}: {v}")

    trials_df = pd.DataFrame(trial_records).sort_values('cv_mae').reset_index(drop=True)
    return study.best_params, study.best_value, trials_df


def run_tuned_5fold(X_selected, y, best_params, kf):
    """최적 파라미터로 선택된 피처셋에서 5-Fold 최종 검증을 수행합니다."""
    print("\n[5/5] 최적 파라미터로 최종 5-Fold K-Fold 교차 검증 수행 중...")
    
    final_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.1,
        'n_estimators': 300,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'subsample_freq': 1,
        **best_params
    }

    oof_preds = np.zeros(len(y))
    fold_maes = []
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y), 1):
        X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_selected.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**final_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        f_mae = mean_absolute_error(y_val, val_pred)
        fold_maes.append(f_mae)
        best_iters.append(model.best_iteration_)
        print(f"      Fold {fold} MAE: {f_mae:.4f} (Best Iteration: {model.best_iteration_})")

    tuned_cv_mae = float(np.mean(fold_maes))
    tuned_oof_mae = float(mean_absolute_error(y, oof_preds))

    return tuned_cv_mae, tuned_oof_mae, fold_maes, best_iters


def export_results(baseline_cv_mae, base_fold_maes, fi_df, selected_features,
                   dropped_features, best_params, tuned_cv_mae, tuned_oof_mae,
                   tuned_fold_maes, best_iters, trials_df):
    """Attempts 폴더에 5th_attempt.xlsx 및 5th_attempt.csv로 결과값을 출력합니다."""
    os.makedirs('Attempts', exist_ok=True)
    excel_path = 'Attempts/5th_attempt.xlsx'
    csv_path = 'Attempts/5th_attempt.csv'

    # 시트 1: 요약 결과 (Summary)
    summary_data = [
        {'항목': '실험 구분', '내용': '5th Attempt: Feature Selection + Optuna Hyperparameter Tuning'},
        {'항목': '검증 방식', '내용': '5-Fold K-Fold 교차 검증 (shuffle=True, random_state=42)'},
        {'항목': '목적 함수 (objective)', '내용': 'regression_l1 (MAE)'},
        {'항목': '평가 지표 (metric)', '내용': 'mae'},
        {'항목': '전체 초기 피처 수', '내용': f"{len(fi_df)}개"},
        {'항목': '제거 피처 수 (하위 10% 및 Gain=0)', '내용': f"{len(dropped_features)}개 ({', '.join(dropped_features)})"},
        {'항목': '최종 선택 피처 수', '내용': f"{len(selected_features)}개"},
        {'항목': '튜닝 전 기준 5-Fold MAE', '내용': f"{baseline_cv_mae:.4f} 초"},
        {'항목': '튜닝 후 최적 5-Fold MAE', '내용': f"{tuned_cv_mae:.4f} 초"},
        {'항목': '튜닝 후 OOF MAE', '내용': f"{tuned_oof_mae:.4f} 초"},
        {'항목': 'MAE 개선량 (변화량)', '내용': f"{baseline_cv_mae - tuned_cv_mae:+.4f} 초 (개선율: {((baseline_cv_mae - tuned_cv_mae) / baseline_cv_mae) * 100:.2f}%)"},
        {'항목': 'Optuna 탐색 횟수 (n_trials)', '내용': '30회'},
    ]
    for param_name, param_val in best_params.items():
        summary_data.append({'항목': f'최적 파라미터: {param_name}', '내용': str(param_val)})
    summary_df = pd.DataFrame(summary_data)

    # 시트 2: 5-Fold 비교 테이블
    folds_data = []
    for f in range(5):
        folds_data.append({
            'Fold': f'Fold {f + 1}',
            '기준 MAE (초)': round(base_fold_maes[f], 4),
            '튜닝 후 MAE (초)': round(tuned_fold_maes[f], 4),
            '변화량 (초)': round(base_fold_maes[f] - tuned_fold_maes[f], 4),
            '튜닝 후 Best Iteration': best_iters[f]
        })
    folds_data.append({
        'Fold': '평균 (Mean)',
        '기준 MAE (초)': round(baseline_cv_mae, 4),
        '튜닝 후 MAE (초)': round(tuned_cv_mae, 4),
        '변화량 (초)': round(baseline_cv_mae - tuned_cv_mae, 4),
        '튜닝 후 Best Iteration': int(round(np.mean(best_iters)))
    })
    folds_df = pd.DataFrame(folds_data)

    # 시트 3: 피처 중요도 및 선택 결과
    fi_export_df = fi_df.copy()
    fi_export_df['선택 여부'] = fi_export_df['feature'].apply(
        lambda x: '제거 (하위 10% 또는 Gain 0)' if x in dropped_features else '선택 (학습에 사용)'
    )

    # 기존 4th_attempt.xlsx의 시트들을 불러와서 보존하면서 5th 시도 정보를 추가
    existing_sheets = {}
    if os.path.exists('Attempts/4th_attempt.xlsx'):
        try:
            xl = pd.ExcelFile('Attempts/4th_attempt.xlsx')
            for s in xl.sheet_names:
                existing_sheets[s] = pd.read_excel('Attempts/4th_attempt.xlsx', sheet_name=s)
        except Exception as e:
            print(f"      [참고] 기존 4th_attempt 읽기 생략 ({e})")

    # 엑셀 파일 저장
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='5th 요약 결과', index=False)
        folds_df.to_excel(writer, sheet_name='5-Fold 검증 비교', index=False)
        fi_export_df.to_excel(writer, sheet_name='피처 중요도 및 선택', index=False)
        trials_df.to_excel(writer, sheet_name='Optuna 튜닝 이력', index=False)

        # 기존 4th_attempt의 시트들도 함께 보존
        for s_name, s_df in existing_sheets.items():
            if s_name not in ['5th 요약 결과', '5-Fold 검증 비교', '피처 중요도 및 선택', 'Optuna 튜닝 이력']:
                # 실험 기록 시트에는 C05 행 추가
                if s_name == '실험 기록':
                    c05_row = pd.DataFrame([{
                        'ID': 'C05-01',
                        '구분': '5-Fold KFold + Optuna 튜닝',
                        '특성 조합': f"선택된 {len(selected_features)}개 피처 (하위 10% 제거)",
                        '모델': 'LightGBM (Optuna 30 trials)',
                        '평일 MAE (월·화·수)': np.nan,
                        '주말 MAE (토)': np.nan,
                        '4일 평균 MAE': tuned_cv_mae,
                        '결정': '5th attempt 최적화 완료',
                        '하이퍼파라미터': str(best_params),
                        'test 평일 (목·금)': np.nan,
                        'test 주말 (일)': np.nan,
                        'test 전체': np.nan,
                        '가설·메모': f"5-Fold 기준 MAE {baseline_cv_mae:.4f}초 -> 튜닝 후 {tuned_cv_mae:.4f}초 (개선: {baseline_cv_mae - tuned_cv_mae:.4f}초)",
                        '날짜': pd.to_datetime('today').strftime('%Y-%m-%d')
                    }])
                    combined_exp = pd.concat([s_df, c05_row], ignore_index=True)
                    combined_exp.to_excel(writer, sheet_name=s_name, index=False)
                else:
                    s_df.to_excel(writer, sheet_name=s_name, index=False)

    # CSV 파일도 요약본으로 함께 저장
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[결과 저장 완료]")
    print(f"   -> 엑셀 파일: {excel_path}")
    print(f"   -> CSV 파일:  {csv_path}")


def main():
    print("=" * 70)
    print(" LightGBM MAE 최적화 파이프라인 (5th Attempt)")
    print(" 1. 5-Fold K-Fold 기준 MAE 및 Gain 피처 중요도 산출")
    print(" 2. Gain 하위 10% 및 0 피처 제거")
    print(" 3. Optuna 30회 튜닝 (num_leaves, min_child_samples, max_depth,")
    print("                    colsample_bytree, subsample, reg_alpha, reg_lambda)")
    print(" 4. 튜닝 전후 검증 MAE 비교 및 Attempts/5th_attempt 결과 저장")
    print("=" * 70)

    # 1. 데이터 로드 및 전처리
    df, X, y, feature_cols, cat_cols = load_and_preprocess_data()

    # 2. 기본 파라미터 5-Fold CV & 피처 중요도
    baseline_cv_mae, base_fold_maes, fi_df, kf = run_baseline_5fold(X, y, feature_cols, cat_cols)

    # 3. 하위 10% 및 Gain=0 피처 제거
    selected_features, dropped_features, kept_df, dropped_df, threshold = select_features(fi_df)
    X_selected = X[selected_features].copy()

    # 4. Optuna 튜닝 (n_trials=30)
    best_params, best_study_mae, trials_df = tune_with_optuna(X_selected, y, kf, n_trials=30)

    # 5. 최적 파라미터 5-Fold 최종 검증
    tuned_cv_mae, tuned_oof_mae, tuned_fold_maes, best_iters = run_tuned_5fold(
        X_selected, y, best_params, kf
    )

    # 6. 결과 파일 내보내기 (Attempts/5th_attempt.xlsx, Attempts/5th_attempt.csv)
    export_results(
        baseline_cv_mae, base_fold_maes, fi_df, selected_features,
        dropped_features, best_params, tuned_cv_mae, tuned_oof_mae,
        tuned_fold_maes, best_iters, trials_df
    )

    # 7. 터미널 요약 출력
    diff = baseline_cv_mae - tuned_cv_mae
    pct = (diff / baseline_cv_mae) * 100.0

    print("\n" + "=" * 70)
    print(" [최종 요약 결과 (Summary)]")
    print("=" * 70)
    print(f" * 튜닝 전 기준 검증 MAE (Baseline 5-Fold CV) : {baseline_cv_mae:.4f} 초")
    print(f" * 튜닝 후 최적 검증 MAE (Tuned 5-Fold CV)    : {tuned_cv_mae:.4f} 초 (OOF MAE: {tuned_oof_mae:.4f} 초)")
    print(f" * 검증 MAE 변화량 (개선값)                  : {diff:+.4f} 초 ({pct:+.2f}%)")
    print("-" * 70)
    print(" * 최종 최적 하이퍼파라미터:")
    for param, val in best_params.items():
        if isinstance(val, float):
            print(f"    - {param:<18}: {val:.6f}" if val < 0.01 else f"    - {param:<18}: {val:.4f}")
        else:
            print(f"    - {param:<18}: {val}")
    print("=" * 70)


if __name__ == '__main__':
    main()
