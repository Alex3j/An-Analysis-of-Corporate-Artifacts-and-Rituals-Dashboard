import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def prepare_weekly_features(df_issues, df_pr, df_comments, maintainers):
    """Готовит недельные агрегации и DataFrame с лагами 1–4 для обучения."""
    # Issues
    closed = df_issues[df_issues['state'] == 'closed'].copy()
    closed['lifetime_days'] = (closed['closed_at'] - closed['created_at']).dt.total_seconds() / (24 * 3600)
    closed['week'] = closed['created_at'].dt.to_period('W').dt.start_time
    weekly_issue = closed.groupby('week')['lifetime_days'].median()

    # PR
    pr_copy = df_pr.copy()
    pr_copy['life_hours'] = (pr_copy['closed_at'] - pr_copy['created_at']).dt.total_seconds() / 3600
    pr_copy['week'] = pr_copy['created_at'].dt.to_period('W').dt.start_time
    weekly_pr = pr_copy.groupby('week')['life_hours'].median()

    # Comments
    df_comments['week'] = df_comments['created_at'].dt.to_period('W').dt.start_time
    neg_all = df_comments.groupby('week')['negativity'].mean()
    neg_maint = df_comments[df_comments['is_maintainer']].groupby('week')['negativity'].mean()

    metrics_map = {
        'median_issue_lifetime_days': ('Время жизни issues (дни)', weekly_issue),
        'median_pr_review_hours': ('Время ревью PR (часы)', weekly_pr),
        'avg_negativity_all': ('Негативность (все)', neg_all),
        'avg_negativity_maintainers': ('Негативность (мейнтейнеры)', neg_maint)
    }

    all_weeks = sorted(set(weekly_issue.index) | set(weekly_pr.index) | set(neg_all.index) | set(neg_maint.index))

    df_weekly = pd.DataFrame({'week': all_weeks})
    for name, (_, series) in metrics_map.items():
        df_weekly[name] = df_weekly['week'].map(series)

    # Лаги 1–4
    for lag in [1, 2, 3, 4]:
        for name in metrics_map.keys():
            df_weekly[f'{name}_lag{lag}'] = df_weekly[name].shift(lag)

    feature_cols = [f'{name}_lag{lag}' for name in metrics_map.keys() for lag in [1, 2, 3, 4]]
    df_model = df_weekly.dropna(subset=feature_cols).copy()

    return df_weekly, df_model, feature_cols, metrics_map


def train_supervised(df_model, feature_cols, crisis_dates):
    """Обучает Random Forest на предкризисных интервалах."""
    y_weekly = pd.Series(0, index=pd.DatetimeIndex(df_model['week']))
    for cd in crisis_dates:
        pre_start = cd - pd.Timedelta(weeks=8)
        pre_end = cd - pd.Timedelta(weeks=4)
        mask = (y_weekly.index >= pre_start) & (y_weekly.index <= pre_end)
        y_weekly[mask] = 1
    y = y_weekly.values

    X = df_model[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, y.sum()


def train_unsupervised(df_model, feature_cols, contamination):
    """Обучает Isolation Forest."""
    X = df_model[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)

    return model, scaler


def evaluate_supervised(model, scaler, df_model, feature_cols):
    """Возвращает вероятность кризиса для последней недели (Supervised)."""
    last_row = df_model.iloc[-1:][feature_cols]
    if last_row.isnull().any().any():
        return None, "Недостаточно данных для оценки последней недели."
    X_last = last_row.values
    X_last_scaled = scaler.transform(X_last)
    proba = model.predict_proba(X_last_scaled)[0][1]
    return proba, None


def evaluate_unsupervised(model, scaler, df_model, feature_cols):
    """Возвращает anomaly score для последней недели (Unsupervised)."""
    last_row = df_model.iloc[-1:][feature_cols]
    if last_row.isnull().any().any():
        return None, "Недостаточно данных для оценки последней недели."
    X_last = last_row.values
    X_last_scaled = scaler.transform(X_last)
    score = model.decision_function(X_last_scaled)[0]
    return score, None


def metric_deviation_details(df_weekly, metrics_map):
    """Сравнение медианы за последние 4 недели с медианой за предшествующие 26 недель."""
    last_week = df_weekly['week'].max()
    recent_4w_mask = (df_weekly['week'] >= last_week - pd.Timedelta(weeks=3)) & (df_weekly['week'] <= last_week)
    recent_4w = df_weekly[recent_4w_mask]
    hist_26w_mask = (df_weekly['week'] >= last_week - pd.Timedelta(weeks=29)) & (df_weekly['week'] < last_week - pd.Timedelta(weeks=3))
    hist_26w = df_weekly[hist_26w_mask]

    alerts = []
    for name, (label, _) in metrics_map.items():
        if recent_4w.empty or recent_4w[name].dropna().empty:
            current_med = np.nan
        else:
            current_med = recent_4w[name].median()

        if hist_26w.empty or hist_26w[name].dropna().empty:
            hist_med = np.nan
        else:
            hist_med = hist_26w[name].median()

        if np.isnan(current_med) or np.isnan(hist_med) or hist_med == 0:
            alerts.append((label, "нет данных", "⚪"))
            continue

        ratio = current_med / hist_med
        if ratio > 5:
            icon, desc = "🔴", f"{current_med:.2f} (в {ratio:.1f} раз выше медианы {hist_med:.2f})"
        elif ratio > 2:
            icon, desc = "🟡", f"{current_med:.2f} (в {ratio:.1f} раз выше медианы {hist_med:.2f})"
        else:
            icon, desc = "🟢", f"{current_med:.2f} (≈ медиана {hist_med:.2f})"
        alerts.append((label, desc, icon))

    return alerts