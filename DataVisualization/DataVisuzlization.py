import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from sklearn.decomposition import PCA
from dash.dependencies import Output, Input

# === Load Data ===
df = pd.read_csv("CSVs/newDataset.csv")

# Identify numeric features
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
features = df[numeric_cols].drop(columns=["anomaly", "train", "sampling", "segment"], errors="ignore")

# PCA Computation
if not features.empty:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

# === Manual Feature Groups ===
feature_groups = {
    "Amplitude Stats":      ["mean", "std", "var"],
    "Shape Stats":          ["kurtosis", "skew"],
    "Peaks":                ["n_peaks", "smooth5_n_peaks", "smooth10_n_peaks", "smooth20_n_peaks"],
    "Derivative Peaks":     ["diff_peaks", "diff2_peaks"],
    "Derivative Variance":  ["diff_var", "diff2_var"],
    "Durations":            ["duration", "len", "len_weighted"],
    "Variance Ratios":      ["var_div_duration", "var_div_len"]
}

# Column descriptions (used in Scatter tab)
column_descriptions = {
    "mean": "Mean (average) value of the segment.",
    "std": "Standard deviation of the segment values.",
    "var": "Variance of the segment values.",
    "kurtosis": "Measure of tailedness of the distribution.",
    "skew": "Skewness, measuring asymmetry of the distribution.",
    "n_peaks": "Number of peaks detected in the signal.",
    "smooth5_n_peaks": "Number of peaks detected after smoothing with a 5-point window.",
    "smooth10_n_peaks": "Number of peaks detected after smoothing with a 10-point window.",
    "smooth20_n_peaks": "Number of peaks detected after smoothing with a 20-point window.",
    "diff_peaks": "Number of peaks detected in the first derivative of the signal.",
    "diff2_peaks": "Number of peaks detected in the second derivative.",
    "diff_var": "Variance of the first derivative.",
    "diff2_var": "Variance of the second derivative.",
    "duration": "Duration of the segment.",
    "len": "Length (number of samples) in the segment.",
    "len_weighted": "Length normalized or weighted feature.",
    "var_div_duration": "Variance normalized by segment duration.",
    "var_div_len": "Variance normalized by segment length."
}

# === Plot Functions ===
def make_scatter(col):
    return px.scatter(
        df.reset_index(), x="index", y=col, color="anomaly",
        title=f"Scatter of {col} by Anomaly",
        labels={"index": "Index", col: col},
        color_continuous_scale=["blue", "red"]
    )

def make_group_box(group_name, log_scale=False):
    """Box plot comparing all features in a group side by side (clean, no raw points)."""
    cols = feature_groups[group_name]
    melted = df[cols].melt(var_name="Feature", value_name="Value")
    fig = px.box(
        melted, x="Feature", y="Value",
        title=f"Box Plot Comparison: {group_name}"
        # no 'points' argument -> clean boxes
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig

# === Static Figures ===
anomaly_fig = px.scatter(
    df.reset_index(), x="index", y="anomaly", color="anomaly",
    title="Anomaly Label Scatter",
    labels={"index": "Index", "anomaly": "Anomaly"},
    color_continuous_scale=["blue", "red"]
)

pca_fig = None
if "PCA1" in df.columns and "PCA2" in df.columns:
    pca_fig = px.scatter(
        df, x="PCA1", y="PCA2", color="anomaly",
        title="PCA of Telemetry Features",
        labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"},
        color_continuous_scale=["blue", "red"]
    )

# === Dash App ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("OPSSAT Anomaly Explorer"),

    dcc.Tabs([

        # ---- Anomaly Overview ----
        dcc.Tab(label="Anomaly Overview", children=[
            dcc.Graph(figure=anomaly_fig)
        ]),

        # ---- PCA Visualization ----
        dcc.Tab(label="PCA Visualization", children=[
            dcc.Graph(figure=pca_fig) if pca_fig else html.Div("No PCA available")
        ]),

        # ---- Grouped Scatter (as before) ----
        dcc.Tab(label="Grouped Scatter", children=[
            html.Label("Select Feature Group:"),
            dcc.Dropdown(
                id="scatter-group",
                options=[{"label": g, "value": g} for g in feature_groups.keys()],
                value=list(feature_groups.keys())[0],
                clearable=False
            ),
            html.Label("Select Feature:"),
            dcc.Dropdown(id="scatter-feature", clearable=False),
            html.Div(id="scatter-description", style={"marginTop": "10px", "fontStyle": "italic"}),
            dcc.Graph(id="scatter-graph")
        ]),

        # ---- Grouped Box Plot (clean) ----
        dcc.Tab(label="Grouped Box Plot", children=[
            html.Label("Select Feature Group:"),
            dcc.Dropdown(
                id="box-group",
                options=[{"label": g, "value": g} for g in feature_groups.keys()],
                value=list(feature_groups.keys())[0],
                clearable=False
            ),
            html.Br(),
            html.Label("Y-axis Scale:"),
            dcc.RadioItems(
                id="box-scale",
                options=[{"label": "Linear", "value": "linear"},
                         {"label": "Log", "value": "log"}],
                value="linear",
                inline=True
            ),
            dcc.Graph(id="box-graph")
        ])
    ])
])

# === Callbacks ===
# Update scatter feature list
@app.callback(
    Output("scatter-feature", "options"),
    Input("scatter-group", "value")
)
def update_scatter_options(group):
    return [{"label": f, "value": f} for f in feature_groups[group]]

# Update scatter plot + description
@app.callback(
    [Output("scatter-graph", "figure"),
     Output("scatter-description", "children")],
    Input("scatter-feature", "value")
)
def update_scatter_plot(feature):
    if feature:
        return make_scatter(feature), column_descriptions.get(feature, "No description available.")
    return {}, ""

# Update grouped box plot
@app.callback(
    Output("box-graph", "figure"),
    [Input("box-group", "value"),
     Input("box-scale", "value")]
)
def update_box_plot(group, scale):
    return make_group_box(group, log_scale=(scale == "log"))

if __name__ == "__main__":
    app.run(debug=True)
