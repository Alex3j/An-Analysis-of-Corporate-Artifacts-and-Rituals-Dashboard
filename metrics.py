import pandas as pd

def find_crisis_dates(df_issues, threshold, target_peak_factor=2):
    """
    Находит даты устойчивого превышения порога открытых issues.
    target_peak_factor – во сколько раз порог должен быть превышен для признания кризиса.
    """
    df_sorted = df_issues.sort_values('created_at')
    open_counts = []
    for date in df_sorted['created_at']:
        open_count = df_sorted[(df_sorted['created_at'] <= date) &
                               (df_sorted['closed_at'].isna() | (df_sorted['closed_at'] > date))].shape[0]
        open_counts.append((date, open_count))
    open_df = pd.DataFrame(open_counts, columns=['date', 'open_count'])

    # Пересечения порога
    crossings = []
    for i in range(1, len(open_df)):
        if open_df.iloc[i-1]['open_count'] < threshold <= open_df.iloc[i]['open_count']:
            crossings.append(open_df.iloc[i]['date'])

    target_peak = threshold * target_peak_factor
    def is_significant(cross_date):
        idx = open_df[open_df['date'] >= cross_date].index[0]
        min_after, max_after = float('inf'), 0
        for j in range(idx, len(open_df)):
            val = open_df.iloc[j]['open_count']
            if val < min_after:
                min_after = val
            if val > max_after:
                max_after = val
            if max_after >= target_peak:
                return min_after >= threshold
            if val < threshold:
                return False
        return max_after == threshold and min_after >= threshold


    crisis_dates = [cd for cd in crossings if is_significant(cd)]
    if not crisis_dates and crossings:
        crisis_dates = [crossings[0]]  # хотя бы первое пересечение
    return open_df, crisis_dates


def calculate_metrics(df_issues, df_pr, df_comments, crisis_date, window_days,
                      maintainers_list=None):
    """Считает метрики 'до/после' для заданного сбоя и окна."""
    date_before = crisis_date - pd.Timedelta(days=window_days)
    date_after = crisis_date + pd.Timedelta(days=window_days)

    # Время жизни закрытых issues
    closed = df_issues[df_issues['state'] == 'closed'].copy()
    closed['lifetime_days'] = (closed['closed_at'] - closed['created_at']).dt.total_seconds() / (24*3600)
    issues_before = closed[closed['closed_at'] < date_before]
    issues_after  = closed[closed['closed_at'] > date_after]
    med_issue_before = issues_before['lifetime_days'].median() if not issues_before.empty else 0
    med_issue_after  = issues_after['lifetime_days'].median() if not issues_after.empty else 0

    # Доля открытых issues
    created_before = df_issues[df_issues['created_at'] <= date_before]
    created_after  = df_issues[df_issues['created_at'] <= date_after]
    open_before = created_before[(created_before['closed_at'].isna()) | (created_before['closed_at'] > date_before)]
    open_after  = created_after[(created_after['closed_at'].isna()) | (created_after['closed_at'] > date_after)]
    ratio_before = open_before.shape[0] / created_before.shape[0] if created_before.shape[0] > 0 else 0
    ratio_after  = open_after.shape[0] / created_after.shape[0] if created_after.shape[0] > 0 else 0

    # Время ревью PR
    pr = df_pr.copy()
    pr['life_hours'] = (pr['closed_at'] - pr['created_at']).dt.total_seconds() / 3600
    pr_before = pr[pr['closed_at'] < date_before]
    pr_after  = pr[pr['closed_at'] > date_after]
    med_rev_before = pr_before['life_hours'].median() if not pr_before.empty else 0
    med_rev_after  = pr_after['life_hours'].median() if not pr_after.empty else 0

    # Негативность комментариев
    comments_before = df_comments[df_comments['created_at'] < crisis_date - pd.Timedelta(days=60)]
    comments_after  = df_comments[df_comments['created_at'] >= crisis_date]
    neg_before = comments_before['negativity'].mean() if not comments_before.empty else 0
    neg_after  = comments_after['negativity'].mean() if not comments_after.empty else 0

    if maintainers_list:
        maint_before = comments_before[comments_before['user'].isin(maintainers_list)]
        maint_after  = comments_after[comments_after['user'].isin(maintainers_list)]
        neg_mb = maint_before['negativity'].mean() if not maint_before.empty else 0
        neg_ma = maint_after['negativity'].mean() if not maint_after.empty else 0
    else:
        neg_mb = neg_ma = 0

    return {
        'median_issue_before': med_issue_before,
        'median_issue_after': med_issue_after,
        'ratio_before': ratio_before,
        'ratio_after': ratio_after,
        'median_review_before': med_rev_before,
        'median_review_after': med_rev_after,
        'neg_before': neg_before,
        'neg_after': neg_after,
        'neg_maint_before': neg_mb,
        'neg_maint_after': neg_ma
    }