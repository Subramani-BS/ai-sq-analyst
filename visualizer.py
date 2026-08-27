import pandas as pd
import plotly.express as px


def auto_visualize(df, question=""):
    if df is None or df.empty:
        return None

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    n_rows = len(df)

    if df.shape == (1, 1):
        return None

    if len(cat_cols) == 1 and len(num_cols) == 1 and n_rows <= 30:
        fig = px.bar(
            df, x=cat_cols[0], y=num_cols[0],
            color=num_cols[0],
            color_continuous_scale="Teal",
            template="plotly_dark"
        )
        fig.update_layout(showlegend=False)
        return fig

    date_cols = [c for c in df.columns if any(
        k in c.lower() for k in ['date', 'time', 'year', 'month']
    )]
    if date_cols and num_cols:
        fig = px.line(
            df, x=date_cols[0], y=num_cols[0],
            template="plotly_dark", markers=True
        )
        return fig

    if len(num_cols) >= 2:
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            template="plotly_dark"
        )
        return fig

    if len(num_cols) == 1 and n_rows > 10:
        fig = px.histogram(
            df, x=num_cols[0],
            template="plotly_dark",
            color_discrete_sequence=["#00b4d8"]
        )
        return fig

    if len(cat_cols) == 1 and len(num_cols) == 1 and n_rows <= 8:
        fig = px.pie(
            df, names=cat_cols[0], values=num_cols[0],
            template="plotly_dark"
        )
        return fig

    return None