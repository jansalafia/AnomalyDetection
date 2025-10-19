import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Neural Network Detection Comparison"

# Paths to CSVs
RESULTS_DIR = "Results/NeuralNetworksResults"
LAST_FILE = os.path.join(RESULTS_DIR, "results_neuralnet.csv")
BEST_FILE = os.path.join(RESULTS_DIR, "results_neuralnet_best.csv")
POISON_FILE = os.path.join(RESULTS_DIR, "results_neuralnet_poisoned.csv")


def load_csv(path):
    """Load a CSV safely, extract accuracy if present."""
    if os.path.exists(path):
        df = pd.read_csv(path).round(3)
        acc = None
        if "accuracy" in df.iloc[:, 0].values:
            row = df[df.iloc[:, 0] == "accuracy"]
            acc = float(row["precision"].values[0])
            df = df[df.iloc[:, 0] != "accuracy"]
        return df.reset_index(drop=True), acc
    else:
        return pd.DataFrame([{"status": "No results found"}]), None


def update_best(last_df, last_acc, best_df, best_acc):
    """Update best results based on last run while preserving accuracy."""
    if best_df.empty or "f1-score" not in best_df.columns:
        # If best doesn't exist, use last as best
        full_best = last_df.copy()
        best_acc = last_acc
    else:
        full_best = best_df.copy()
        for idx in last_df.index:
            if idx < len(best_df):
                if last_df.loc[idx, "f1-score"] > best_df.loc[idx, "f1-score"]:
                    full_best.loc[idx] = last_df.loc[idx]
    # Save best results with accuracy row preserved
    if best_acc is not None:
        acc_row = pd.DataFrame([{
            full_best.columns[0]: "accuracy",
            "precision": best_acc,
            **{col: "" for col in full_best.columns[2:]}
        }])
        full_best = pd.concat([acc_row, full_best], ignore_index=True)
    full_best.to_csv(BEST_FILE, index=False)
    return full_best, best_acc


def load_results():
    last_df, last_acc = load_csv(LAST_FILE)
    best_df, best_acc = load_csv(BEST_FILE)
    poisoned_df, poisoned_acc = load_csv(POISON_FILE)

    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc)

    return {
        "Last Result": (last_df, last_acc),
        "Best Result": (best_df, best_acc),
        "Poisoned Data Result": (poisoned_df, poisoned_acc)
    }


app.layout = html.Div([
    html.H1("Neural Network Detection Results", style={"textAlign": "center"}),

    dcc.Interval(id="interval-refresh", interval=5000, n_intervals=0),

    html.Div(id="tables-container", style={
        "display": "flex",
        "gap": "20px",
        "alignItems": "flex-start"
    })
])


@app.callback(Output("tables-container", "children"),
              Input("interval-refresh", "n_intervals"))
def refresh_tables(_):
    results = load_results()
    cards = []

    for title, (df, acc) in results.items():
        # Filter out accuracy row from table display
        display_df = df[df.iloc[:, 0] != "accuracy"].reset_index(drop=True)

        accuracy_display = (
            html.H2(f"Accuracy: {acc:.3f}", style={"color": "#0074D9", "textAlign": "center"})
            if acc is not None else html.H2("Accuracy: —", style={"color": "gray", "textAlign": "center"})
        )

        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in display_df.columns],
            data=display_df.to_dict("records"),
            style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold", "textAlign": "center"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                {"if": {"state": "active"}, "backgroundColor": "#e6f2ff", "border": "1px solid #0074D9"}
            ]
        )

        card = html.Div([
            html.H3(title, style={"textAlign": "center", "marginBottom": "10px"}),
            accuracy_display,
            table
        ], style={"flex": "1", "backgroundColor": "white", "padding": "10px",
                  "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                  "border": "1px solid #ddd"})

        cards.append(card)

    return cards


if __name__ == "__main__":
    app.run(debug=True)