import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def auto_visualize(df, question=""):
    if df is None or df.empty:
        return None

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    n_rows = len(df)
    question_lower = question.lower()

    # Single value result
    if df.shape == (1, 1):
        return None

    # ── Detect chart type from question keywords ──────────
    wants_pie = any(w in question_lower for w in
                    ['pie', 'share', 'percentage', 'proportion', 'distribution'])
    wants_line = any(w in question_lower for w in
                     ['trend', 'over time', 'monthly', 'yearly', 'daily',
                      'weekly', 'time series', 'growth'])
    wants_bar = any(w in question_lower for w in
                    ['compare', 'comparison', 'top', 'best', 'worst',
                     'highest', 'lowest', 'ranking', 'by region', 'by category'])
    wants_scatter = any(w in question_lower for w in
                        ['correlation', 'relationship', 'vs', 'versus',
                         'scatter', 'between'])
    wants_hist = any(w in question_lower for w in
                     ['distribution', 'spread', 'histogram', 'frequency'])

    # ── Keyword-based chart selection ─────────────────────

    # Pie chart
    if wants_pie and len(cat_cols) >= 1 and len(num_cols) >= 1:
        fig = px.pie(
            df, names=cat_cols[0], values=num_cols[0],
            title=f"{num_cols[0]} by {cat_cols[0]}",
            template="plotly_dark",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    # Line chart
    if wants_line:
        date_cols = [c for c in df.columns if any(
            k in c.lower() for k in ['date', 'time', 'year', 'month', 'day', 'week']
        )]
        x_col = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else None)
        if x_col and num_cols:
            fig = px.line(
                df, x=x_col, y=num_cols[0],
                title=f"{num_cols[0]} over {x_col}",
                template="plotly_dark",
                markers=True,
                line_shape="spline"
            )
            fig.update_traces(line_color="#00b4d8", line_width=2.5)
            return fig

    # Bar chart
    if wants_bar and cat_cols and num_cols:
        fig = px.bar(
            df, x=cat_cols[0], y=num_cols[0],
            title=f"{num_cols[0]} by {cat_cols[0]}",
            template="plotly_dark",
            color=num_cols[0],
            color_continuous_scale="Teal",
            text=num_cols[0]
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(showlegend=False)
        return fig

    # Scatter
    if wants_scatter and len(num_cols) >= 2:
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            title=f"{num_cols[1]} vs {num_cols[0]}",
            template="plotly_dark",
            trendline="ols" if n_rows > 5 else None,
            color_discrete_sequence=["#00b4d8"]
        )
        return fig

    # Histogram
    if wants_hist and num_cols:
        fig = px.histogram(
            df, x=num_cols[0],
            title=f"Distribution of {num_cols[0]}",
            template="plotly_dark",
            color_discrete_sequence=["#00b4d8"],
            nbins=30
        )
        return fig

    # ── Auto detect based on data shape ──────────────────

    # Date + number → Line
    date_cols = [c for c in df.columns if any(
        k in c.lower() for k in ['date', 'time', 'year', 'month', 'day']
    )]
    if date_cols and num_cols:
        fig = px.line(
            df, x=date_cols[0], y=num_cols[0],
            title=f"{num_cols[0]} over {date_cols[0]}",
            template="plotly_dark",
            markers=True
        )
        fig.update_traces(line_color="#00b4d8")
        return fig

    # Category + number → Bar
    if len(cat_cols) == 1 and len(num_cols) == 1 and n_rows <= 30:
        fig = px.bar(
            df, x=cat_cols[0], y=num_cols[0],
            title=f"{num_cols[0]} by {cat_cols[0]}",
            template="plotly_dark",
            color=num_cols[0],
            color_continuous_scale="Teal",
            text=num_cols[0]
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(showlegend=False)
        return fig

    # 2 numeric → Scatter
    if len(num_cols) >= 2:
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            title=f"{num_cols[1]} vs {num_cols[0]}",
            template="plotly_dark",
            color_discrete_sequence=["#00b4d8"]
        )
        return fig

    # Single numeric many rows → Histogram
    if len(num_cols) == 1 and n_rows > 10:
        fig = px.histogram(
            df, x=num_cols[0],
            title=f"Distribution of {num_cols[0]}",
            template="plotly_dark",
            color_discrete_sequence=["#00b4d8"]
        )
        return fig

    # Small category + number → Pie
    if len(cat_cols) == 1 and len(num_cols) == 1 and n_rows <= 8:
        fig = px.pie(
            df, names=cat_cols[0], values=num_cols[0],
            title=f"{num_cols[0]} by {cat_cols[0]}",
            template="plotly_dark",
            hole=0.3
        )
        return fig

    return None