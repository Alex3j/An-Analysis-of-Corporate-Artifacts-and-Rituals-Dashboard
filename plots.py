import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_open_issues_trend(open_df, crisis_dates, threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=open_df['date'], y=open_df['open_count'],
                             mode='lines', name='Открытые issues',
                             line=dict(color='red', width=2)))
    fig.add_hline(y=threshold, line_dash="dot", line_color="grey",
                  annotation_text=f"Порог {threshold}", annotation_position="bottom right")
    for cd in crisis_dates:
        # Линия сбоя
        fig.add_shape(
            type='line', x0=cd, x1=cd, y0=0, y1=1,
            line=dict(dash='dash', color='blue'),
            xref='x', yref='paper'
        )
        # Подпись над линией
        fig.add_annotation(
            x=cd, y=1, xref='x', yref='paper',
            text=f"Сбой {cd.date()}", showarrow=False,
            yshift=10, font=dict(color='blue')
        )
    fig.update_layout(title='Накопление открытых issues',
                      xaxis_title='Дата', yaxis_title='Количество')
    return fig

def plot_weekly_median_issue_lifetime(weekly_data):
    fig = px.line(weekly_data, x='week', y='lifetime_days', markers=True,
                  labels={'lifetime_days': 'Медианное время жизни (дни)'})
    fig.update_layout(title='Медианное время закрытия issues по неделям')
    return fig

def plot_weekly_median_review_time(weekly_data):
    fig = px.line(weekly_data, x='week', y='life_hours', markers=True,
                  labels={'life_hours': 'Медианное время ревью (часы)'})
    fig.update_layout(title='Медианное время ревью PR по неделям')
    return fig

def plot_weekly_negativity(weekly_neg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_neg.index, y=weekly_neg['all'],
                             mode='lines+markers', name='Все комментарии'))
    fig.add_trace(go.Scatter(x=weekly_neg.index, y=weekly_neg['maintainers'],
                             mode='lines+markers', name='Мейнтейнеры'))
    fig.update_layout(title='Средняя негативность комментариев (VADER)',
                      xaxis_title='Неделя', yaxis_title='Индекс негативности')
    return fig