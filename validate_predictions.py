import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. Подготовка недельных данных ----------
def prepare_weekly_features(df_issues, df_pr, df_comments):
    closed = df_issues[df_issues['state'] == 'closed'].copy()
    closed['lifetime_days'] = (closed['closed_at'] - closed['created_at']).dt.total_seconds() / (24*3600)
    closed['week'] = closed['created_at'].dt.to_period('W').dt.start_time
    weekly_issue = closed.groupby('week')['lifetime_days'].median()

    pr_copy = df_pr.copy()
    pr_copy['life_hours'] = (pr_copy['closed_at'] - pr_copy['created_at']).dt.total_seconds() / 3600
    pr_copy['week'] = pr_copy['created_at'].dt.to_period('W').dt.start_time
    weekly_pr = pr_copy.groupby('week')['life_hours'].median()

    df_comments['week'] = df_comments['created_at'].dt.to_period('W').dt.start_time
    neg_all = df_comments.groupby('week')['negativity'].mean()

    metrics_map = {
        'median_issue_lifetime_days': weekly_issue,
        'median_pr_review_hours': weekly_pr,
        'avg_negativity_all': neg_all
    }

    all_weeks = sorted(set(weekly_issue.index) | set(weekly_pr.index) | set(neg_all.index))
    df_weekly = pd.DataFrame({'week': all_weeks})
    for name, series in metrics_map.items():
        df_weekly[name] = df_weekly['week'].map(series)

    for lag in [1, 2, 3, 4]:
        for name in metrics_map.keys():
            df_weekly[f'{name}_lag{lag}'] = df_weekly[name].shift(lag)

    feature_cols = [f'{name}_lag{lag}' for name in metrics_map.keys() for lag in [1, 2, 3, 4]]
    df_model = df_weekly.dropna(subset=feature_cols).copy()
    return df_weekly, df_model, feature_cols

# ---------- 2. Загрузка lodash ----------
issues = pd.read_csv('data/lodash_issues.csv', parse_dates=['created_at', 'closed_at'])
pr = pd.read_csv('data/lodash_pr.csv', parse_dates=['created_at', 'closed_at'])
comments = pd.read_csv('data/lodash_comments.csv', parse_dates=['created_at'])

for col in ['created_at', 'closed_at']:
    if col in issues.columns:
        issues[col] = pd.to_datetime(issues[col], utc=True).dt.tz_localize(None)
for col in ['created_at', 'closed_at', 'merged_at']:
    if col in pr.columns:
        pr[col] = pd.to_datetime(pr[col], utc=True).dt.tz_localize(None)
if 'created_at' in comments.columns:
    comments['created_at'] = pd.to_datetime(comments['created_at'], utc=True).dt.tz_localize(None)

analyzer = SentimentIntensityAnalyzer()
comments['negativity'] = comments['body'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['neg'])

df_weekly, df_model, feature_cols = prepare_weekly_features(issues, pr, comments)

# ---------- 3. Параметры кризиса ----------
crisis_date = pd.Timestamp('2020-04-17')
pre_start = crisis_date - pd.Timedelta(weeks=8)
pre_end = crisis_date - pd.Timedelta(weeks=4)

# ---------- 4. Supervised ----------
y_weekly = pd.Series(0, index=pd.DatetimeIndex(df_weekly['week']))
mask = (y_weekly.index >= pre_start) & (y_weekly.index <= pre_end)
y_weekly[mask] = 1
df_model['crisis_label'] = y_weekly.loc[df_model['week']].values

X_all = df_model[feature_cols].values
y_all = df_model['crisis_label'].values

scaler_clf = StandardScaler()
X_all_scaled = scaler_clf.fit_transform(X_all)

model_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_clf.fit(X_all_scaled, y_all)
probas = model_clf.predict_proba(X_all_scaled)[:, 1]

# ---------- 5. Unsupervised Isolation Forest ----------
scaler_if = StandardScaler()
X_all_scaled_if = scaler_if.fit_transform(X_all)
model_if = IsolationForest(contamination=0.05, random_state=42)
model_if.fit(X_all_scaled_if)
anomaly_scores = model_if.decision_function(X_all_scaled_if)
anomaly_labels = model_if.predict(X_all_scaled_if)

# ---------- 6. Графики ----------
# Supervised
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_model['week'], y=probas, mode='lines+markers', name='Вероятность кризиса'))
fig1.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Порог 0.7")
fig1.add_hline(y=0.3, line_dash="dot", line_color="orange", annotation_text="Порог 0.3")

# Вертикальная линия сбоя – ручная, чтобы избежать ошибки Timestamp
fig1.add_shape(type='line', x0=crisis_date, x1=crisis_date, y0=0, y1=1,
               line=dict(dash='dash', color='black'), xref='x', yref='paper')
fig1.add_annotation(x=crisis_date, y=1, xref='x', yref='paper',
                    text="Сбой 2020-04-17", showarrow=False, yshift=10,
                    font=dict(color='black'))

fig1.update_layout(title="Supervised: вероятность кризиса по неделям (lodash)",
                   xaxis_title="Неделя", yaxis_title="Вероятность")
fig1.write_html('supervised_validation_lodash.html')
print("Supervised график сохранён в supervised_validation_lodash.html")

# Unsupervised
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_model['week'], y=anomaly_scores, mode='lines+markers', name='Anomaly score'))
fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Порог 0")

fig2.add_shape(type='line', x0=crisis_date, x1=crisis_date, y0=0, y1=1,
               line=dict(dash='dash', color='black'), xref='x', yref='paper')
fig2.add_annotation(x=crisis_date, y=1, xref='x', yref='paper',
                    text="Сбой 2020-04-17", showarrow=False, yshift=10,
                    font=dict(color='black'))

fig2.update_layout(title="Unsupervised: anomaly score по неделям (lodash)",
                   xaxis_title="Неделя", yaxis_title="Anomaly score")
fig2.write_html('unsupervised_validation_lodash.html')
print("Unsupervised график сохранён в unsupervised_validation_lodash.html")

# ---------- 7. Оценка ----------
pre_mask = (df_model['week'] >= pre_start) & (df_model['week'] <= pre_end)
pre_probas = probas[pre_mask]
if len(pre_probas) > 0:
    print(f"Supervised: средняя вероятность в предкризисном окне: {pre_probas.mean():.1%}")

pre_idx = np.where(pre_mask)[0]
if len(pre_idx) > 0:
    pre_anomalies = (anomaly_labels[pre_idx] == -1).mean()
    print(f"Unsupervised: доля аномальных недель в предкризисном окне: {pre_anomalies:.1%}")