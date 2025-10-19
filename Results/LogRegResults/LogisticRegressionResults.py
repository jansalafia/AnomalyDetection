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

# 🧩 Define all poisoning result files here
POISONING_FILES = {
    "2% Label Flip Poisoning": os.path.join(RESULTS_DIR, "results_logreg_poisoned_2.csv"),
    "10% Label Flip Poisoning": os.path.join(RESULTS_DIR, "results_logreg_poisoned_10.csv")
}


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
        full_best = last_df.copy()
        best_acc = last_acc
    else:
        full_best = best_df.copy()
        for idx in last_df.index:
            if idx < len(best_df):
                if last_df.loc[idx, "f1-score"] > best_df.loc[idx, "f1-score"]:
                    full_best.loc[idx] = last_df.loc[idx]
    # Reattach accuracy and save
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
    """Load all result files, including poisoning variants."""
    last_df, last_acc = load_csv(LAST_FILE)
    best_df, best_acc = load_csv(BEST_FILE)

    # Update the best run
    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc)

    # Base results
    results = {
        "Last Run": (last_df, last_acc),
        "Best Run": (best_df, best_acc),
    }

    # Load all poisoning results dynamically
    for attack_name, file_path in POISONING_FILES.items():
        df, acc = load_csv(file_path)
        results[attack_name] = (df, acc)

    return results


app.layout = html.Div([
    html.H1("Logistic Regression Results", style={"textAlign": "center"}),

    dcc.Interval(id="interval-refresh", interval=10 * 1000, n_intervals=0),  # Auto-refresh every 10s

    html.Div(id="tables-container", style={
        "display": "flex",
        "flexWrap": "wrap",
        "justifyContent": "center",
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
            "width": "45%",
            "backgroundColor": "white"
        })
        cards.append(card)

    return cards


if __name__ == "__main__":
    app.run(debug=True)
