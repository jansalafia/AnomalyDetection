import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Universal Model Comparison Dashboard"

# Define all models and their result directories
MODELS = {
    "Logistic Regression": "Results/LogRegResults",
    "Neural Network": "Results/NeuralNetworksResults",
    "Random Forest": "Results/RandomForestResults",
    "SVM": "Results/SVMResults",
    "XGBoost": "Results/XGBoostResults",
}


# ----------- Helper Functions -----------

def load_csv(path):
    """Load CSV safely, extract accuracy if present."""
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


def update_best(last_df, last_acc, best_df, best_acc, best_path):
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
    # Save best results with accuracy row
    if best_acc is not None:
        acc_row = pd.DataFrame([{
            full_best.columns[0]: "accuracy",
            "precision": best_acc,
            **{col: "" for col in full_best.columns[2:]}
        }])
        full_best = pd.concat([acc_row, full_best], ignore_index=True)
    full_best.to_csv(best_path, index=False)
    return full_best, best_acc


def load_model_results(model_name, folder):
    """Load all result files for a given model (including poisoned)."""
    # Standard file names
    last_file = os.path.join(folder, f"results_{model_name.lower().replace(' ', '')}.csv")
    best_file = os.path.join(folder, f"results_{model_name.lower().replace(' ', '')}_best.csv")

    # Load main results
    last_df, last_acc = load_csv(last_file)
    best_df, best_acc = load_csv(best_file)

    # Update best
    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc, best_file)

    results = {
        "Last Result": (last_df, last_acc),
        "Best Result": (best_df, best_acc),
    }

    # Load all poisoning files dynamically
    for file in os.listdir(folder):
        if "poisoned" in file and file.endswith(".csv"):
            label = file.replace("results_", "").replace(".csv", "").replace("_", " ").title()
            df, acc = load_csv(os.path.join(folder, file))
            results[label] = (df, acc)

    return results


# ----------- Layout -----------

app.layout = html.Div([
    html.H1("Universal Detection Model Comparison", style={"textAlign": "center"}),

    # Tabs for models
    dcc.Tabs(
        id="model-tabs",
        value=list(MODELS.keys())[0],
        children=[
            dcc.Tab(label=model, value=model)
            for model in MODELS.keys()
        ],
        style={"marginBottom": "20px"}
    ),

    dcc.Interval(id="interval-refresh", interval=5 * 1000, n_intervals=0),

    html.Div(id="tables-container", style={
        "display": "flex",
        "flexWrap": "wrap",
        "justifyContent": "center",
        "gap": "20px"
    })
])


# ----------- Callback -----------

@app.callback(
    Output("tables-container", "children"),
    Input("interval-refresh", "n_intervals"),
    Input("model-tabs", "value")
)
def update_tables(_, selected_model):
    folder = MODELS[selected_model]
    results = load_model_results(selected_model, folder)
    cards = []

    for title, (df, acc) in results.items():
        # Filter out accuracy row for display
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
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                {"if": {"state": "active"}, "backgroundColor": "#e6f2ff", "border": "1px solid #0074D9"}
            ]
        )

        card = html.Div([
            html.H3(title, style={"textAlign": "center", "marginBottom": "10px"}),
            accuracy_display,
            table
        ], style={
            "flex": "1",
            "backgroundColor": "white",
            "padding": "10px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            "border": "1px solid #ddd",
            "minWidth": "350px",
            "maxWidth": "45%"
        })

        cards.append(card)

    return cards


# ----------- Run -----------

if __name__ == "__main__":
    app.run(debug=True)
