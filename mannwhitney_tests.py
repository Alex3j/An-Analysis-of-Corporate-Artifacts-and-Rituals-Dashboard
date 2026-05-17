# mannwhitney_tests.py
import pandas as pd
from scipy.stats import mannwhitneyu
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ----------------- Загрузка данных -----------------
def load_data(project):
    issues = pd.read_csv(f'data/{project}_issues.csv', parse_dates=['created_at', 'closed_at'])
    pr = pd.read_csv(f'data/{project}_pr.csv', parse_dates=['created_at', 'closed_at'])
    comments = pd.read_csv(f'data/{project}_comments.csv', parse_dates=['created_at'])
    for col in ['created_at', 'closed_at']:
        if col in issues.columns:
            issues[col] = pd.to_datetime(issues[col], utc=True).dt.tz_localize(None)
    for col in ['created_at', 'closed_at', 'merged_at']:
        if col in pr.columns:
            pr[col] = pd.to_datetime(pr[col], utc=True).dt.tz_localize(None)
    if 'created_at' in comments.columns:
        comments['created_at'] = pd.to_datetime(comments['created_at'], utc=True).dt.tz_localize(None)
    return issues, pr, comments

# ----------------- Подготовка метрик -----------------
def add_metrics(issues, pr, comments, maintainers_list=None):
    # Issues
    closed_issues = issues[issues['state'] == 'closed'].copy()
    closed_issues['lifetime_days'] = (closed_issues['closed_at'] - closed_issues['created_at']).dt.total_seconds() / (24*3600)

    # PR
    pr_copy = pr.copy()
    pr_copy['life_hours'] = (pr_copy['closed_at'] - pr_copy['created_at']).dt.total_seconds() / 3600

    # Comments negativity
    analyzer = SentimentIntensityAnalyzer()
    comments_copy = comments.copy()
    comments_copy['negativity'] = comments_copy['body'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['neg'])

    if maintainers_list:
        comments_copy['is_maintainer'] = comments_copy['user'].isin(maintainers_list)
    else:
        comments_copy['is_maintainer'] = False

    return closed_issues, pr_copy, comments_copy

# ----------------- Фильтрация по окну -----------------
def filter_samples(df_issues, df_pr, df_comments, crisis_date, window_days, maintainers=False):
    date_before = crisis_date - pd.Timedelta(days=window_days)
    date_after = crisis_date + pd.Timedelta(days=window_days)

    # Issues lifetime
    issues_before = df_issues[df_issues['closed_at'] < date_before]['lifetime_days'].dropna().values
    issues_after = df_issues[df_issues['closed_at'] > date_after]['lifetime_days'].dropna().values

    # PR review hours
    pr_before = df_pr[df_pr['closed_at'] < date_before]['life_hours'].dropna().values
    pr_after = df_pr[df_pr['closed_at'] > date_after]['life_hours'].dropna().values

    # Comments negativity
    comments_before = df_comments[df_comments['created_at'] < crisis_date - pd.Timedelta(days=60)]
    comments_after = df_comments[df_comments['created_at'] >= crisis_date]

    if maintainers:
        comments_before = comments_before[comments_before['is_maintainer']]
        comments_after = comments_after[comments_after['is_maintainer']]

    neg_before = comments_before['negativity'].dropna().values
    neg_after = comments_after['negativity'].dropna().values

    return {
        'issue_lifetime': (issues_before, issues_after),
        'pr_review': (pr_before, pr_after),
        'negativity': (neg_before, neg_after)
    }

# ----------------- Применение U-теста -----------------
def run_tests(samples_dict):
    results = {}
    for key, (before, after) in samples_dict.items():
        if len(before) == 0 or len(after) == 0:
            results[key] = None
            continue
        try:
            stat, p = mannwhitneyu(before, after, alternative='two-sided')
            results[key] = p
        except Exception:
            results[key] = None
    return results

# ================ ГЛАВНЫЙ БЛОК =================
# Конфигурация сбоев
crises = {
    'lodash': pd.Timestamp('2020-04-17'),
    'axios1': pd.Timestamp('2018-05-08'),
    'axios2': pd.Timestamp('2022-10-08')
}
windows = [60, 365]

# Мейнтейнеры (вручную, как в работе)
maintainers_lodash = ['jdalton', 'jridgewell', 'megawac', 'bnjmnt4n', 'phated']
maintainers_axios = ['jasonsaayman', 'DigitalBrainJS', 'mzabriskie', 'nickuraltsev']

# Загрузка данных
lodash_issues, lodash_pr, lodash_comments = load_data('lodash')
axios_issues, axios_pr, axios_comments = load_data('axios')

# Добавление метрик
lodash_issues_m, lodash_pr_m, lodash_comments_m = add_metrics(lodash_issues, lodash_pr, lodash_comments, maintainers_lodash)
axios_issues_m, axios_pr_m, axios_comments_m = add_metrics(axios_issues, axios_pr, axios_comments, maintainers_axios)

# Словарь для хранения p-value
results_table = []

# Перебор всех сбоев и окон
for crisis_key, crisis_date in crises.items():
    if crisis_key == 'lodash':
        issues_m = lodash_issues_m
        pr_m = lodash_pr_m
        comments_m = lodash_comments_m
    else:
        issues_m = axios_issues_m
        pr_m = axios_pr_m
        comments_m = axios_comments_m

    for window_days in windows:
        samples_all = filter_samples(issues_m, pr_m, comments_m, crisis_date, window_days, maintainers=False)
        samples_maint = filter_samples(issues_m, pr_m, comments_m, crisis_date, window_days, maintainers=True)

        p_values_all = run_tests(samples_all)
        p_values_maint = run_tests(samples_maint)

        results_table.append({
            'Crisis': crisis_key,
            'Window': window_days,
            'Issue lifetime p': p_values_all.get('issue_lifetime'),
            'PR review p': p_values_all.get('pr_review'),
            'Negativity all p': p_values_all.get('negativity'),
            'Negativity maintainers p': p_values_maint.get('negativity')
        })

# Вывод таблицы
df_results = pd.DataFrame(results_table)
print("\n=== P-значения U-критерия Манна-Уитни ===")
print(df_results.to_string(index=False))

# Дополнительно: размеры выборок
print("\n=== Размеры выборок (до / после) ===")
for crisis_key, crisis_date in crises.items():
    if crisis_key == 'lodash':
        issues_m = lodash_issues_m
        pr_m = lodash_pr_m
        comments_m = lodash_comments_m
    else:
        issues_m = axios_issues_m
        pr_m = axios_pr_m
        comments_m = axios_comments_m

    for window_days in windows:
        samples_all = filter_samples(issues_m, pr_m, comments_m, crisis_date, window_days, maintainers=False)
        print(f"\n{crisis_key} (окно {window_days}):")
        for key in ['issue_lifetime', 'pr_review', 'negativity']:
            before, after = samples_all[key]
            print(f"  {key}: до={len(before)}, после={len(after)}")