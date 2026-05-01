import streamlit as st
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from collector import GitHubCollector
from data_loader import load_csv_data
from metrics import find_crisis_dates, calculate_metrics
from plots import (plot_open_issues_trend, plot_weekly_median_issue_lifetime,
                   plot_weekly_median_review_time, plot_weekly_negativity)

st.set_page_config(page_title="Анализ здоровья ИТ-команды", layout="wide")
st.title("Дашборд: Ранние индикаторы сбоев и выгорания")

# ---------- Инициализация сессионного состояния ----------
if 'data' not in st.session_state:
    st.session_state.data = None
if 'project_name' not in st.session_state:
    st.session_state.project_name = ""

# ---------- Боковая панель: выбор источника данных ----------
st.sidebar.header("1. Источник данных")
source = st.sidebar.radio("Выберите источник", ["Локальные CSV", "GitHub API"])

if source == "Локальные CSV":
    project_name = st.sidebar.text_input("Префикс файлов (например, 'lodash' или 'axios')", value="lodash")
    if st.sidebar.button("Загрузить CSV"):
        try:
            data = load_csv_data(project_name, folder='data')
            st.session_state.data = data
            st.session_state.project_name = project_name
            st.success(f"Данные проекта '{project_name}' загружены")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
else:
    token = st.sidebar.text_input("GitHub токен (с правами repo)", type="password")
    repo_full = st.sidebar.text_input("Репозиторий (например, 'axios/axios')", value="lodash/lodash")
    if st.sidebar.button("Собрать данные через API"):
        if not token:
            st.sidebar.warning("Введите токен")
        else:
            with st.status("Сбор данных...", expanded=True) as status:
                try:
                    collector = GitHubCollector(token, repo_full)
                            
                    status.update(label="Собираю issues...")
                    df_issues = collector.collect_issues(
                        save_csv=True,
                        status_callback=lambda msg: st.write(msg)
                    )
                            
                    status.update(label="Собираю pull requests...")
                    df_pr = collector.collect_prs(
                        save_csv=True,
                        status_callback=lambda msg: st.write(msg)
                    )
                            
                    issue_numbers = df_issues['number'].tolist()
                    pr_numbers = df_pr['number'].tolist()
                            
                    status.update(label="Собираю комментарии (может занять время)...")
                    df_comments = collector.collect_all_comments(
                        issue_numbers, pr_numbers,
                        save_csv=True,
                            status_callback=lambda msg: st.write(msg)
                    )
                            
                    data = {'issues': df_issues, 'pr': df_pr, 'comments': df_comments}
                    st.session_state.data = data
                    st.session_state.project_name = repo_full.replace('/', '_')
                    status.update(label="Сбор данных завершён!", state="complete")
                except Exception as e:
                    status.update(label="Ошибка сбора данных", state="error")
                    st.error(f"Ошибка: {e}")

# Если данные загружены – продолжаем
if st.session_state.data is not None:
    data = st.session_state.data
    df_issues = data['issues']
    df_pr = data['pr']
    df_comments = data['comments']

    # ---------- Анализ тональности (один раз) ----------
    if 'negativity' not in df_comments.columns:
        analyzer = SentimentIntensityAnalyzer()
        df_comments['negativity'] = df_comments['body'].astype(str).apply(
            lambda x: analyzer.polarity_scores(x)['neg'])

    # ---------- Боковая панель: мейнтейнеры ----------
    st.sidebar.header("2. Мейнтейнеры")
    auto_maint = st.sidebar.checkbox("Определить автоматически (топ-5 по комментариям)", value=True)
    if auto_maint:
        # Исключаем ники, содержащие '[bot]'
        human_comments = df_comments[~df_comments['user'].str.contains(r'\[bot\]', case=False, na=False)]
        top_users = human_comments['user'].value_counts().head(5).index.tolist()
        maintainers = top_users
        st.sidebar.write("Автоопределённые мейнтейнеры:", ", ".join(maintainers))
    else:
        maint_input = st.sidebar.text_input("Введите ники через запятую",
                                            value="jdalton, jridgewell" if st.session_state.project_name.startswith('lodash')
                                            else "jasonsaayman, DigitalBrainJS")
        maintainers = [m.strip() for m in maint_input.split(',') if m.strip()]

    df_comments['is_maintainer'] = df_comments['user'].isin(maintainers)

    # ---------- Боковая панель: параметры сбоя ----------
    st.sidebar.header("3. Параметры анализа сбоев")
    threshold = st.sidebar.slider("Порог открытых issues", min_value=10, max_value=500, value=50)
    target_peak_factor = st.sidebar.slider("Коэффициент пика кризиса", min_value=1.5, max_value=5.0, value=2.0, step=0.5,
                                           help="Во сколько раз порог должен быть превышен, чтобы кризис считался значимым")

    # Пересчёт графика накопления и дат сбоев
    open_df, crisis_dates = find_crisis_dates(df_issues, threshold, target_peak_factor)

    # ---------- Основная область ----------
    tab1, tab2, tab3 = st.tabs(["📈 Обзор проекта", "⚠️ Метрики по сбоям", "📅 Динамика по неделям"])

    with tab1:
        st.subheader("Накопление открытых issues с порогом сбоя")
        fig_open = plot_open_issues_trend(open_df, crisis_dates, threshold)
        st.plotly_chart(fig_open, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего issues", len(df_issues))
        col2.metric("Открыто сейчас", df_issues[df_issues['state'] == 'open'].shape[0])
        col3.metric("Значимых сбоев", len(crisis_dates))
        if crisis_dates:
            st.write("**Даты сбоев:**")
            for i, cd in enumerate(crisis_dates, 1):
                st.write(f"{i}. {cd.date()}")

    with tab2:
        st.subheader("Метрики до/после сбоя")
        if crisis_dates:
            selected_window = st.radio("Размер окна (дни)", [60, 365], horizontal=True)
            selected_crisis = st.selectbox("Выберите сбой", [cd.date() for cd in crisis_dates])
            crisis_date = pd.Timestamp(selected_crisis)

            metrics = calculate_metrics(df_issues, df_pr, df_comments,
                                        crisis_date, selected_window, maintainers)

            # Метрики с инверсными цветами: рост -> красный, снижение -> зелёный
            col1, col2, col3 = st.columns(3)
            col1.metric("Время жизни issues (дни)",
                        f"{metrics['median_issue_after']:.1f}",
                        f"{metrics['median_issue_after'] - metrics['median_issue_before']:.1f}",
                        delta_color="inverse")
            col2.metric("Доля открытых issues",
                        f"{metrics['ratio_after']:.1%}",
                        f"{metrics['ratio_after'] - metrics['ratio_before']:.1%}",
                        delta_color="inverse")
            col3.metric("Время ревью PR (часы)",
                        f"{metrics['median_review_after']:.1f}",
                        f"{metrics['median_review_after'] - metrics['median_review_before']:.1f}",
                        delta_color="inverse")

            col4, col5 = st.columns(2)
            col4.metric("Негативность (все)",
                        f"{metrics['neg_after']:.3f}",
                        f"{metrics['neg_after'] - metrics['neg_before']:.3f}",
                        delta_color="inverse")
            col5.metric("Негативность (мейнтейнеры)",
                        f"{metrics['neg_maint_after']:.3f}",
                        f"{metrics['neg_maint_after'] - metrics['neg_maint_before']:.3f}",
                        delta_color="inverse")

            with st.expander("Сравнительная таблица"):
                df_comp = pd.DataFrame({
                    'Показатель': ['Время жизни issues', 'Доля открытых issues', 'Время ревью PR',
                                   'Негативность (все)', 'Негативность (мейнтейнеры)'],
                    'До': [f"{metrics['median_issue_before']:.1f}",
                           f"{metrics['ratio_before']:.1%}",
                           f"{metrics['median_review_before']:.1f}",
                           f"{metrics['neg_before']:.3f}",
                           f"{metrics['neg_maint_before']:.3f}"],
                    'После': [f"{metrics['median_issue_after']:.1f}",
                              f"{metrics['ratio_after']:.1%}",
                              f"{metrics['median_review_after']:.1f}",
                              f"{metrics['neg_after']:.3f}",
                              f"{metrics['neg_maint_after']:.3f}"]
                }).set_index('Показатель')
                st.table(df_comp)
        else:
            st.info("Не найдено значимых сбоев. Попробуйте уменьшить порог или коэффициент пика.")

    with tab3:
        st.subheader("Недельные тренды")
        # Подготовка weekly агрегаций
        closed = df_issues[df_issues['state'] == 'closed'].copy()
        closed['lifetime_days'] = (closed['closed_at'] - closed['created_at']).dt.total_seconds() / (24*3600)
        closed['week'] = closed['created_at'].dt.to_period('W').dt.start_time
        weekly_issue = closed.groupby('week')['lifetime_days'].median().reset_index(name='lifetime_days')

        pr_copy = df_pr.copy()
        pr_copy['life_hours'] = (pr_copy['closed_at'] - pr_copy['created_at']).dt.total_seconds() / 3600
        pr_copy['week'] = pr_copy['created_at'].dt.to_period('W').dt.start_time
        weekly_pr = pr_copy.groupby('week')['life_hours'].median().reset_index(name='life_hours')

        df_comments['week'] = df_comments['created_at'].dt.to_period('W').dt.start_time
        neg_all = df_comments.groupby('week')['negativity'].mean()
        neg_maint = df_comments[df_comments['is_maintainer']].groupby('week')['negativity'].mean()
        weekly_neg = pd.DataFrame({'all': neg_all, 'maintainers': neg_maint})

        st.plotly_chart(plot_weekly_median_issue_lifetime(weekly_issue), use_container_width=True)
        st.plotly_chart(plot_weekly_median_review_time(weekly_pr), use_container_width=True)
        st.plotly_chart(plot_weekly_negativity(weekly_neg), use_container_width=True)

        # Сборка данных для экспорта
        export_df = weekly_issue.merge(weekly_pr, on='week', how='outer')
        export_df = export_df.merge(weekly_neg, on='week', how='outer')
        export_df = export_df.rename(columns={
            'lifetime_days': 'median_issue_lifetime_days',
            'life_hours': 'median_pr_review_hours',
            'all': 'avg_negativity_all',
            'maintainers': 'avg_negativity_maintainers'
        })
        export_csv = export_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Скачать недельные данные (CSV)",
            data=export_csv,
            file_name=f"{st.session_state.project_name}_weekly_metrics.csv",
            mime="text/csv"
        )
else:
    st.info("Загрузите данные (CSV или через API), чтобы начать анализ.")