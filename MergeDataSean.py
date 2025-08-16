import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import os

# === File paths ===
input_folder = "ToBeMerged"
output_folder = "Output"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Input files
file_summary = os.path.join(input_folder, "dataset.csv")
file_segments = os.path.join(input_folder, "segments.csv")

# Output merged file
output_file = os.path.join(output_folder, "merged_OPSSAT_segments.csv")

# === Load datasets ===
df_summary = pd.read_csv(file_summary)
df_segments = pd.read_csv(file_segments)

# === Merge datasets ===
merged_df = pd.merge(
    df_segments,
    df_summary,
    on=["segment", "channel", "anomaly", "train", "sampling"],
    how="left"
)

# Convert timestamps to datetime
merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])

# === Save merged dataset ===
merged_df.to_csv(output_file, index=False)
print(f"Merged dataset saved to {output_file}")

# === Dash App ===
app = dash.Dash(__name__)

# Dropdown options = list of all unique segments
segment_options = [{"label": f"Segment {s}", "value": s} for s in sorted(merged_df["segment"].unique())]

app.layout = html.Div([
    html.H1("OPSSAT Segment Explorer"),

    dcc.Dropdown(
        id="segment-dropdown",
        options=segment_options,
        value=merged_df["segment"].iloc[0],  # default = first segment
        clearable=False
    ),

    dcc.Graph(id="time-series-plot"),

    html.Div(id="summary-stats", style={"marginTop": "20px"})
])

# === Callbacks ===
@app.callback(
    [dash.dependencies.Output("time-series-plot", "figure"),
     dash.dependencies.Output("summary-stats", "children")],
    [dash.dependencies.Input("segment-dropdown", "value")]
)
def update_plot(segment_id):
    seg_data = merged_df[merged_df["segment"] == segment_id]
    seg_summary = df_summary[df_summary["segment"] == segment_id].iloc[0]

    # Time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seg_data["timestamp"],
        y=seg_data["value"],
        mode="lines",
        name="Raw values"
    ))
    fig.update_layout(
        title=f"Segment {segment_id} - Channel {seg_summary['channel']}",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white"
    )

    # Summary stats table
    summary_html = html.Table([
        html.Tr([html.Th(col), html.Td(str(seg_summary[col]))]) for col in df_summary.columns
    ], style={"border": "1px solid black", "borderCollapse": "collapse"})

    return fig, summary_html

if __name__ == "__main__":
    app.run(debug=True)
