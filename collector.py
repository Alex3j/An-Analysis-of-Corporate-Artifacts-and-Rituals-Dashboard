import pandas as pd
import time
from github import Github

class GitHubCollector:
    def __init__(self, token: str, repo_name: str):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)
        self.repo_name = repo_name

    def _log(self, message, status_callback=None):
        if status_callback:
            status_callback(message)
        else:
            print(message)

    def collect_issues(self, save_csv=True, status_callback=None) -> pd.DataFrame:
        issues = []
        for issue in self.repo.get_issues(state='all', sort='created', direction='desc'):
            if issue.pull_request is None:
                issues.append({
                    'number': issue.number,
                    'title': issue.title,
                    'created_at': issue.created_at,
                    'closed_at': issue.closed_at,
                    'state': issue.state,
                    'comments': issue.comments,
                    'labels': [l.name for l in issue.labels],
                    'body': issue.body[:500] if issue.body else ''
                })
                if len(issues) % 100 == 0:
                    self._log(f"Собрано {len(issues)} issues...", status_callback)
                time.sleep(1)
        self._log(f"Итого issues: {len(issues)}", status_callback)
        df = pd.DataFrame(issues)
        if save_csv:
            df.to_csv(f'data/{self.repo_name.split("/")[-1]}_issues.csv', index=False)
        return df

    def collect_prs(self, save_csv=True, status_callback=None) -> pd.DataFrame:
        prs = []
        for pr in self.repo.get_pulls(state='all', sort='created', direction='desc'):
            prs.append({
                'number': pr.number,
                'created_at': pr.created_at,
                'closed_at': pr.closed_at,
                'merged_at': pr.merged_at,
                'user': pr.user.login,
                'state': pr.state,
                'comments': pr.comments
            })
            if len(prs) % 50 == 0:
                self._log(f"Собрано {len(prs)} PR...", status_callback)
            time.sleep(0.5)
        self._log(f"Итого PR: {len(prs)}", status_callback)
        df = pd.DataFrame(prs)
        if save_csv:
            df.to_csv(f'data/{self.repo_name.split("/")[-1]}_pr.csv', index=False)
        return df

    def collect_all_comments(self, issue_numbers, pr_numbers=None, save_csv=True, status_callback=None) -> pd.DataFrame:
        comments = []
        # Комментарии к issues
        total_issues = len(issue_numbers)
        for idx, num in enumerate(issue_numbers):
            try:
                issue = self.repo.get_issue(num)
                for comment in issue.get_comments():
                    comments.append({
                        'number': num,
                        'type': 'issue',
                        'user': comment.user.login,
                        'created_at': comment.created_at,
                        'body': comment.body or ''
                    })
                time.sleep(0.3)
                if idx % 100 == 0:
                    self._log(f"Комментарии issues: {idx}/{total_issues}, всего комментариев: {len(comments)}", status_callback)
            except Exception as e:
                self._log(f"Ошибка issue #{num}: {e}", status_callback)

        # Комментарии к PR
        if pr_numbers is not None:
            total_prs = len(pr_numbers)
            for idx, pr_num in enumerate(pr_numbers):
                try:
                    pr = self.repo.get_pull(pr_num)
                    for comment in pr.get_issue_comments():
                        comments.append({
                            'number': pr_num,
                            'type': 'pr',
                            'user': comment.user.login,
                            'created_at': comment.created_at,
                            'body': comment.body or ''
                        })
                    time.sleep(0.3)
                    if idx % 50 == 0:
                        self._log(f"Комментарии PR: {idx}/{total_prs}, всего комментариев: {len(comments)}", status_callback)
                except Exception as e:
                    self._log(f"Ошибка PR #{pr_num}: {e}", status_callback)

        self._log(f"Итого комментариев: {len(comments)}", status_callback)
        df = pd.DataFrame(comments)
        if save_csv:
            df.to_csv(f'data/{self.repo_name.split("/")[-1]}_comments.csv', index=False)
        return df