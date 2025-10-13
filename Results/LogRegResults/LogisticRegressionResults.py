import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash(__name__)
app.title = "Logistic Regression Results Comparison"

RESULTS_DIR = "Results/LogRegResults"
LAST_FILE = os.path.join(RESULTS_DIR, "results_logreg.csv")
BEST_FILE = os.path.join(RESULTS_DIR, "results_logreg_best.csv")
POISON_FILE = os.path.join(RESULTS_DIR, "results_logreg_poisoned.csv")


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
    # Save best results
    # Reattach accuracy row at top
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
        "Last Run": (last_df, last_acc),
        "Best Run": (best_df, best_acc),
        "Poisoned Data": (poisoned_df, poisoned_acc)
    }


app.layout = html.Div([
    html.H1("Logistic Regression Results", style={"textAlign": "center"}),

    dcc.Interval(id="interval-refresh", interval=10*1000, n_intervals=0),  # Auto-refresh every 10s

    html.Div(id="tables-container", style={
        "display": "flex",
        "justifyContent": "space-around",
        "alignItems": "flex-start",
        "gap": "20px",
        "marginTop": "30px"
    })
])


@app.callback(Output("tables-container", "children"),
              Input("interval-refresh", "n_intervals"))
def update_tables(_):
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
            style_table={"overflowX": "auto", "minWidth": "300px"},
            style_cell={"textAlign": "center", "padding": "5px"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
            ]
        )

        card = html.Div([
            html.H3(title, style={"textAlign": "center"}),
            accuracy_display,
            table
        ], style={
            "border": "2px solid #ddd",
            "borderRadius": "10px",
            "padding": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            "width": "32%",
            "backgroundColor": "white"
        })
        cards.append(card)

    return cards



if __name__ == "__main__":
    app.run(debug=True)
