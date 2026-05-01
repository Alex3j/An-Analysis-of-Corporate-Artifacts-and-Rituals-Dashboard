import pandas as pd

def load_csv_data(project_prefix, folder='data'):
    """Загружает CSV-файлы проекта по префиксу (например, 'lodash' или 'axios')."""
    import os
    path = os.path.join(folder, project_prefix)
    issues = pd.read_csv(f'{path}_issues.csv', parse_dates=['created_at', 'closed_at'])
    pr = pd.read_csv(f'{path}_pr.csv', parse_dates=['created_at', 'closed_at', 'merged_at'])
    comments = pd.read_csv(f'{path}_comments.csv', parse_dates=['created_at'])

    # Приведение дат к UTC и удаление часового пояса
    for col in ['created_at', 'closed_at']:
        if col in issues.columns:
            issues[col] = pd.to_datetime(issues[col], utc=True).dt.tz_localize(None)
    for col in ['created_at', 'closed_at', 'merged_at']:
        if col in pr.columns:
            pr[col] = pd.to_datetime(pr[col], utc=True).dt.tz_localize(None)
    if 'created_at' in comments.columns:
        comments['created_at'] = pd.to_datetime(comments['created_at'], utc=True).dt.tz_localize(None)

    return {'issues': issues, 'pr': pr, 'comments': comments}