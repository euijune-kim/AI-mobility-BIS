"""
Generate Final Submission using 6th Attempt Best Pipeline (5-Fold Multi-Seed LightGBM)
Produces row_id and tt_sec submission CSV.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print(" 최고 성능 모델 (6th Attempt) 기반 최종 Test 예측 CSV 생성")
    print("=" * 70)

    # 1. 데이터 로드
    train_path = 'data/final/train.parquet' if os.path.exists('data/final/train.parquet') else 'data/final/train.csv'
    test_path = 'data/final/test.parquet' if os.path.exists('data/final/test.parquet') else 'data/final/test.csv'

    print(f"[1/4] 데이터셋 로드: Train={train_path}, Test={test_path}")
    train = pd.read_parquet(train_path) if train_path.endswith('.parquet') else pd.read_csv(train_path)
    test = pd.read_parquet(test_path) if test_path.endswith('.parquet') else pd.read_csv(test_path)
    print(f"      Train 크기: {train.shape}, Test 크기: {test.shape}")

    # 2. 피처 엔지니어링 (공통)
    print("[2/4] 도메인 피처 생성 중...")
    PEAK_HOURS = {7, 8, 9, 17, 18, 19}

    for df in [train, test]:
        dt = pd.to_datetime(df['date'])
        df['dow'] = dt.dt.day_of_week
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        df['is_peak'] = df['arrive_hour'].isin(PEAK_HOURS).astype(int)
        df['interval'] = np.select(
            [df['dow'] == 5, df['dow'] == 6],
            [df['interval_sat'], df['interval_sun']],
            default=df['interval_wd']
        )
        df['hour_sin'] = np.sin(2 * np.pi * df['arrive_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['arrive_hour'] / 24)

        df['dlat'] = df['to_lat'] - df['from_lat']
        df['dlon'] = df['to_lon'] - df['from_lon']
        df['manhattan_dist'] = np.abs(df['dlat']) * 111000 + np.abs(df['dlon']) * 88800
        df['log_dist'] = np.log1p(df['dist_m'])
        df['segment_id'] = df['from_stop'].astype(str) + '_' + df['to_stop'].astype(str)
        df['route_dir'] = df['route_id'].astype(str) + '_' + df['direction'].astype(str)

        # 결합 키
        df['seg_weekend'] = df['segment_id'] + '_' + df['is_weekend'].astype(str)
        df['seg_peak'] = df['segment_id'] + '_' + df['is_peak'].astype(str)
        df['seg_hour'] = df['segment_id'] + '_' + df['arrive_hour'].astype(str)

    # 노선 내 상대 위치 (Train 기준으로 노선별 max_seq 계산)
    max_seq_map = train.groupby('route_id')['stop_seq'].max().to_dict()
    train['rel_stop_seq'] = train['stop_seq'] / train['route_id'].map(max_seq_map)
    test['rel_stop_seq'] = test['stop_seq'] / test['route_id'].map(max_seq_map).fillna(train['stop_seq'].max())

    drop_cols = ['row_id', 'date', 'tt_sec', 'seg_weekend', 'seg_peak', 'seg_hour']
    cat_cols = ['route_id', 'route_nm', 'vehicle_id', 'from_stop', 'from_name', 'to_stop', 'to_name', 'segment_id', 'route_dir']

    # 3. 5-Fold K-Fold 및 Test 예측치 누적
    print("[3/4] 5-Fold Multi-Seed (Seeds: 42, 2024) 교차 검증 및 Test 예측 시작...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_maes = []
    seeds = [42, 2024]

    for fold, (train_idx, val_idx) in enumerate(kf.split(train), 1):
        X_tr = train.iloc[train_idx].copy()
        y_tr = train['tt_sec'].iloc[train_idx].copy()
        X_va = train.iloc[val_idx].copy()
        y_val = train['tt_sec'].iloc[val_idx].copy()
        X_te = test.copy()

        global_med = y_tr.median()

        # OOF Target Aggregations
        seg_med = X_tr.groupby('segment_id')['tt_sec'].median().to_dict()
        X_tr['seg_med'] = X_tr['segment_id'].map(seg_med).fillna(global_med)
        X_va['seg_med'] = X_va['segment_id'].map(seg_med).fillna(global_med)
        X_te['seg_med'] = X_te['segment_id'].map(seg_med).fillna(global_med)

        seg_wk_med = X_tr.groupby('seg_weekend')['tt_sec'].median().to_dict()
        X_tr['seg_wk_med'] = X_tr['seg_weekend'].map(seg_wk_med).fillna(X_tr['seg_med'])
        X_va['seg_wk_med'] = X_va['seg_weekend'].map(seg_wk_med).fillna(X_va['seg_med'])
        X_te['seg_wk_med'] = X_te['seg_weekend'].map(seg_wk_med).fillna(X_te['seg_med'])

        seg_pk_med = X_tr.groupby('seg_peak')['tt_sec'].median().to_dict()
        X_tr['seg_pk_med'] = X_tr['seg_peak'].map(seg_pk_med).fillna(X_tr['seg_med'])
        X_va['seg_pk_med'] = X_va['seg_peak'].map(seg_pk_med).fillna(X_va['seg_med'])
        X_te['seg_pk_med'] = X_te['seg_peak'].map(seg_pk_med).fillna(X_te['seg_med'])

        seg_h_med = X_tr.groupby('seg_hour')['tt_sec'].median().to_dict()
        X_tr['seg_h_med'] = X_tr['seg_hour'].map(seg_h_med).fillna(X_tr['seg_pk_med'])
        X_va['seg_h_med'] = X_va['seg_hour'].map(seg_h_med).fillna(X_va['seg_pk_med'])
        X_te['seg_h_med'] = X_te['seg_hour'].map(seg_h_med).fillna(X_te['seg_pk_med'])

        rhw_med = X_tr.groupby(['route_id', 'arrive_hour', 'is_weekend'])['tt_sec'].median().to_dict()
        X_tr['rhw_med'] = X_tr.set_index(['route_id', 'arrive_hour', 'is_weekend']).index.map(rhw_med).fillna(global_med)
        X_va['rhw_med'] = X_va.set_index(['route_id', 'arrive_hour', 'is_weekend']).index.map(rhw_med).fillna(global_med)
        X_te['rhw_med'] = X_te.set_index(['route_id', 'arrive_hour', 'is_weekend']).index.map(rhw_med).fillna(global_med)

        # 동적 피처 계산
        for data_split in [X_tr, X_va, X_te]:
            data_split['exp_speed_wk'] = data_split['dist_m'] / (data_split['seg_wk_med'] + 1.0)
            data_split['exp_speed_h'] = data_split['dist_m'] / (data_split['seg_h_med'] + 1.0)
            data_split['delay_wk'] = data_split['seg_wk_med'] - data_split['seg_med']
            data_split['delay_h'] = data_split['seg_h_med'] - data_split['seg_med']

        features = [c for c in X_tr.columns if c not in drop_cols]
        for c in cat_cols:
            if c in features:
                X_tr[c] = X_tr[c].astype('category')
                X_va[c] = X_va[c].astype('category')
                X_te[c] = X_te[c].astype('category')

        fold_val_pred = np.zeros(len(val_idx))
        fold_test_pred = np.zeros(len(test))

        for seed in seeds:
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'importance_type': 'gain',
                'max_depth': 12,
                'num_leaves': 255,
                'min_child_samples': 22,
                'colsample_bytree': 0.55,
                'subsample': 0.95,
                'subsample_freq': 1,
                'extra_trees': True,
                'path_smooth': 1.0,
                'learning_rate': 0.035,
                'n_estimators': 1500,
                'random_state': seed,
                'n_jobs': -1,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[features], y_tr,
                eval_set=[(X_va[features], y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            val_p = model.predict(X_va[features])
            test_p = model.predict(X_te[features])

            fold_val_pred += val_p / len(seeds)
            fold_test_pred += test_p / len(seeds)

        oof_preds[val_idx] = fold_val_pred
        test_preds += fold_test_pred / 5.0
        f_mae = mean_absolute_error(y_val, fold_val_pred)
        fold_maes.append(f_mae)
        print(f"      Fold {fold}/5 MAE: {f_mae:.4f}")

    mean_cv_mae = np.mean(fold_maes)
    print(f"\n   >>> 5-Fold 최종 평균 MAE: {mean_cv_mae:.4f} 초")

    # 4. 제출 파일 생성 및 유효성 검사
    print("\n[4/4] 최종 제출 CSV 파일 저장 및 검증...")
    # 물리적 최소값 4.0초 클리핑 및 반올림
    final_pred = np.clip(test_preds, 4.0, 1800.0).round(1)

    sub_df = pd.DataFrame({
        'row_id': test['row_id'],
        'tt_sec': final_pred
    })

    # 유효성 검사
    assert len(sub_df) == 111563, f"행 수 불일치: {len(sub_df)}"
    assert sub_df['tt_sec'].isna().sum() == 0, "결측값 존재"
    assert sub_df['row_id'].duplicated().sum() == 0, "row_id 중복 존재"

    # 파일 저장 (루트 및 submissions 폴더)
    os.makedirs('submissions', exist_ok=True)
    out_csv_root = 'final_submission.csv'
    out_csv_sub = 'submissions/final_6th_attempt.csv'

    sub_df.to_csv(out_csv_root, index=False)
    sub_df.to_csv(out_csv_sub, index=False)

    print(f"   -> 루트 CSV 저장 완료: {out_csv_root}")
    print(f"   -> submissions CSV 저장 완료: {out_csv_sub}")
    print("   -> 상위 5행:")
    print(sub_df.head(5).to_string(index=False))
    print("   -> 예측값 통계 요약:")
    print(sub_df['tt_sec'].describe().to_string())


if __name__ == '__main__':
    main()
