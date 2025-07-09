import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
import io
import base64

# Load data
df = pd.read_csv("data/CombinedUnemplpymentData.csv", parse_dates=["Date"])
df["MonthNum"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Initialize app
app = dash.Dash(__name__)
app.title = "Unemployment EDA Dashboard"

# Layout
app.layout = html.Div([
    html.H1("U.S. Unemployment by Age Group"),

    dcc.Dropdown(
        id="agegroup-filter",
        options=[{"label": grp, "value": grp} for grp in sorted(df["AgeGroup"].unique())],
        value=sorted(df["AgeGroup"].unique()),
        multi=True
    ),

    dcc.DatePickerRange(
        id="date-range",
        start_date=df["Date"].min(),
        end_date=df["Date"].max(),
        display_format="YYYY-MM"
    ),

    dcc.Checklist(
        id="smooth-toggle",
        options=[{"label": "Apply Rolling Average", "value": "smooth"}],
        value=[]
    ),

    dcc.Graph(id="time-series"),
    dcc.Graph(id="forecast-series"),

    html.Hr(),
    html.H3("Monthly Trend by Age Group"),
    dcc.Dropdown(
        id="monthly-agegroup-filter",
        options=[{"label": grp, "value": grp} for grp in sorted(df["AgeGroup"].unique())],
        value=sorted(df["AgeGroup"].unique()),
        multi=True
    ),
    dcc.Graph(id="monthly-trend"),

    html.H3("Year-over-Year Change by Age Group"),
    dcc.Dropdown(
        id="yoy-agegroup-filter",
        options=[{"label": grp, "value": grp} for grp in sorted(df["AgeGroup"].unique())],
        value=sorted(df["AgeGroup"].unique()),
        multi=True
    ),
    dcc.Graph(id="yoy-trend"),

    html.H3("YOY Heatmap"),
    dcc.Graph(id="heatmap")

], style={"width": "85%", "margin": "auto"})

# Callbacks
@app.callback(
    Output("time-series", "figure"),
    Input("agegroup-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("smooth-toggle", "value")
)
def update_time_series(selected, start_date, end_date, smooth):
    filtered = df[(df["AgeGroup"].isin(selected)) &
                  (df["Date"] >= pd.to_datetime(start_date)) &
                  (df["Date"] <= pd.to_datetime(end_date))]

    if "smooth" in smooth:
        filtered["Unemployment"] = filtered.groupby("AgeGroup")["Unemployment"]\
            .transform(lambda x: x.rolling(3, min_periods=1).mean())

    fig = px.line(filtered, x="Date", y="Unemployment", color="AgeGroup",
                  title="Unemployment Over Time",
                  hover_data={"Date": True, "Unemployment": ".2f", "AgeGroup": True})
    return fig

@app.callback(
    Output("forecast-series", "figure"),
    Input("agegroup-filter", "value")
)
def update_forecast(selected):
    temp = df[df["AgeGroup"].isin(selected)]
    fig = go.Figure()
    for group in selected:
        group_df = temp[temp["AgeGroup"] == group].sort_values("Date")
        ts = group_df.groupby("Date")["Unemployment"].mean().asfreq("MS")

        ts = ts.interpolate()
        try:
            model = ExponentialSmoothing(ts, seasonal="add", seasonal_periods=12)
            fit = model.fit()
            forecast = fit.forecast(12)
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name=f"{group} Forecast"))
        except Exception as e:
            continue
    fig.update_layout(title="Forecasted Unemployment (Next 12 Months)", xaxis_title="Date")
    return fig

@app.callback(
    Output("monthly-trend", "figure"),
    Input("monthly-agegroup-filter", "value")
)
def update_monthly(selected):
    temp = df[df["AgeGroup"].isin(selected)]
    monthly_avg = temp.groupby(["MonthNum", "AgeGroup"])["Unemployment"].mean().reset_index()
    fig = px.line(monthly_avg, x="MonthNum", y="Unemployment", color="AgeGroup",
                  title="Average Monthly Unemployment",
                  hover_data={"MonthNum": True, "Unemployment": ".2f"})
    fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13))))
    return fig

@app.callback(
    Output("yoy-trend", "figure"),
    Input("yoy-agegroup-filter", "value")
)
def update_yoy(selected):
    temp = df[df["AgeGroup"].isin(selected)]
    yearly_avg = temp.groupby(["Year", "AgeGroup"])["Unemployment"].mean().unstack()
    yoy = yearly_avg.diff()

    fig = go.Figure()
    for group in selected:
        if group in yoy.columns:
            fig.add_trace(go.Scatter(
                x=yoy.index, y=yoy[group], mode="lines+markers", name=group,
                hovertemplate=f"%{{x}}<br>{group}: %{{y:.2f}}"
            ))
    fig.update_layout(title="Year-over-Year Change in Unemployment", xaxis_title="Year")
    return fig

@app.callback(
    Output("heatmap", "figure"),
    Input("yoy-agegroup-filter", "value")
)
def update_heatmap(selected):
    temp = df[df["AgeGroup"].isin(selected)]
    yearly_avg = temp.groupby(["Year", "AgeGroup"])["Unemployment"].mean().unstack()
    yoy = yearly_avg.diff().T  # Transposed to match sns heatmap style

    fig = go.Figure(data=go.Heatmap(
        z=yoy.values,
        x=yoy.columns,
        y=yoy.index,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="YOY Change")
    ))
    fig.update_layout(
        title="YOY Change Heatmap by Age Group",
        xaxis_title="Year",
        yaxis_title="Age Group"
    )
    return fig

# Run app
if __name__ == "__main__":
    app.run(debug=True)

















