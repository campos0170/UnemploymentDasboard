import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SARIMAX
from prophet import Prophet
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

    dcc.Dropdown(
        id="model-selector",
        options=[
            {"label": "Exponential Smoothing", "value": "exp"},
            {"label": "ARIMA", "value": "arima"},
            {"label": "Prophet", "value": "prophet"}
        ],
        value="exp",
        style={"marginTop": "10px"}
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
    Input("agegroup-filter", "value"),
    Input("model-selector", "value")
)
def update_forecast(selected, model_type):
    temp = df[df["AgeGroup"].isin(selected)]
    fig = go.Figure()
    for group in selected:
        group_df = temp[temp["AgeGroup"] == group].sort_values("Date")
        ts = group_df.groupby("Date")["Unemployment"].mean().asfreq("MS")
        ts = ts.interpolate()

        try:
            if model_type == "exp":
                model = ExponentialSmoothing(ts, seasonal="add", seasonal_periods=12)
                fit = model.fit()
                forecast = fit.forecast(12)
                ci_upper = forecast + ts.std()
                ci_lower = forecast - ts.std()
                fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"{group} Actual"))
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name=f"{group} Forecast"))
                fig.add_trace(go.Scatter(x=forecast.index, y=ci_upper, mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Upper CI", showlegend=False))
                fig.add_trace(go.Scatter(x=forecast.index, y=ci_lower, mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Lower CI", fill='tonexty', showlegend=False))
            elif model_type == "arima":
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                fit = model.fit(disp=False)
                forecast = fit.get_forecast(steps=12)
                pred = forecast.predicted_mean
                ci = forecast.conf_int()
                fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"{group} Actual"))
                fig.add_trace(go.Scatter(x=pred.index, y=pred, mode="lines", name=f"{group} Forecast"))
                fig.add_trace(go.Scatter(x=pred.index, y=ci.iloc[:, 0], mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Lower CI", showlegend=False))
                fig.add_trace(go.Scatter(x=pred.index, y=ci.iloc[:, 1], mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Upper CI", fill='tonexty', showlegend=False))
            elif model_type == "prophet":
                prophet_df = group_df[["Date", "Unemployment"]].rename(columns={"Date": "ds", "Unemployment": "y"})
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=12, freq='M')
                forecast = m.predict(future)
                fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name=f"{group} Actual"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"{group} Forecast"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Upper CI", showlegend=False))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(dash='dash'),
                                         name=f"{group} Lower CI", fill='tonexty', showlegend=False))
        except Exception as e:
            continue

    fig.update_layout(title="Forecasted Unemployment (Next 12 Months)", xaxis_title="Date", yaxis_title="Unemployment Rate")
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
    yoy = yearly_avg.diff().T

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

















