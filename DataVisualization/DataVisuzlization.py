import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv("CSVs/OPSAT-AD_modified.csv")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
features = df[numeric_cols].drop(columns=['anomaly', 'train', 'sampling', 'segment'], errors='ignore')

# PCA computation
if not features.empty:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

# === Column descriptions ===
column_descriptions = {
    "segment": "Identifier for the data segment.",
    "anomaly": "Binary label indicating whether the segment is anomalous (1) or normal (0).",
    "train": "Indicator if the segment belongs to the training set.",
    "sampling": "Sampling rate or method used.",
    "duration": "Duration of the segment.",
    "len": "Length (number of samples) in the segment.",
    "mean": "Mean (average) value of the segment.",
    "var": "Variance of the segment values.",
    "std": "Standard deviation of the segment values.",
    "kurtosis": "Measure of tailedness of the distribution.",
    "skew": "Skewness, measuring asymmetry of the distribution.",
    "n_peaks": "Number of peaks detected in the signal.",
    "smooth10_n_peaks": "Number of peaks detected after smoothing with a 10-point window.",
    "smooth20_n_peaks": "Number of peaks detected after smoothing with a 20-point window.",
    "diff_peaks": "Number of peaks detected in the first derivative of the signal.",
    "diff2_peaks": "Number of peaks detected in the second derivative.",
    "diff_var": "Variance of the first derivative.",
    "diff2_var": "Variance of the second derivative.",
    "gaps_squared": "Sum of squared gaps between peaks.",
    "len_weighted": "Length normalized or weighted feature.",
    "var_div_duration": "Variance normalized by segment duration.",
    "var_div_len": "Variance normalized by segment length."
}

# === Dash App ===
app = dash.Dash(__name__)

# Tab 1: Anomaly overview
anomaly_fig = px.scatter(df.reset_index(), x="index", y="anomaly", color="anomaly",
                         title="Anomaly Label Scatter",
                         labels={"index": "Index", "anomaly": "Anomaly"},
                         color_continuous_scale=["blue", "red"])

# Tab 2: PCA Scatter
pca_fig = None
if 'PCA1' in df.columns and 'PCA2' in df.columns:
    pca_fig = px.scatter(df, x="PCA1", y="PCA2", color="anomaly",
                         title="PCA of Telemetry Features",
                         labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"},
                         color_continuous_scale=["blue", "red"])

# Tab 3: Per-column scatter plots
column_options = [col for col in features.columns]

def make_column_plot(col):
    return px.scatter(df.reset_index(), x="index", y=col, color="anomaly",
                      title=f"Scatter of {col} by Anomaly",
                      labels={"index": "Index", col: col},
                      color_continuous_scale=["blue", "red"])

app.layout = html.Div([
    html.H1("OPSSAT Anomaly Explorer"),

    dcc.Tabs([
        dcc.Tab(label="Anomaly Overview", children=[
            dcc.Graph(figure=anomaly_fig),
        ]),

        dcc.Tab(label="PCA Visualization", children=[
            dcc.Graph(figure=pca_fig) if pca_fig else html.Div("No PCA available")
        ]),

        dcc.Tab(label="Per-Column Distributions", children=[
            html.Label("Select Column:"),
            dcc.Dropdown(
                id="column-dropdown",
                options=[{"label": c, "value": c} for c in column_options],
                value=column_options[0] if column_options else None,
                clearable=False
            ),
            html.Div(id="column-description", style={"marginTop": "10px", "fontStyle": "italic"}),
            dcc.Graph(id="column-graph")
        ])
    ])
])

# === Callbacks ===
@app.callback(
    [dash.dependencies.Output("column-graph", "figure"),
     dash.dependencies.Output("column-description", "children")],
    [dash.dependencies.Input("column-dropdown", "value")]
)
def update_column_plot(selected_col):
    if selected_col:
        fig = make_column_plot(selected_col)
        description = column_descriptions.get(selected_col, "No description available for this column.")
        return fig, description
    return {}, ""

if __name__ == "__main__":
    app.run(debug=True)
