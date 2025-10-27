import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objs as go

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Universal Model Comparison Dashboard"

# Define models and directories
MODELS = {
    "Logistic Regression": "Results/LogRegResults",
    "Neural Network": "Results/NeuralNetworksResults",
    "Random Forest": "Results/RandomForestResults",
    "SVM": "Results/SVMResults",
    "XGBoost": "Results/XGBoostResults",
}

# -------- Helper Functions --------
def load_csv(path):
    """Safely load a CSV and extract accuracy if present."""
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
    """Update best results based on F1-score improvement."""
    if best_df.empty or "f1-score" not in best_df.columns:
        full_best = last_df.copy()
        best_acc = last_acc
    else:
        full_best = best_df.copy()
        for idx in last_df.index:
            if idx < len(best_df):
                if last_df.loc[idx, "f1-score"] > best_df.loc[idx, "f1-score"]:
                    full_best.loc[idx] = last_df.loc[idx]

    # Save best file with accuracy row
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
    """Load results for a given model, including poisoning files."""
    last_file = os.path.join(folder, f"results_{model_name.lower().replace(' ', '')}.csv")
    best_file = os.path.join(folder, f"results_{model_name.lower().replace(' ', '')}_best.csv")

    last_df, last_acc = load_csv(last_file)
    best_df, best_acc = load_csv(best_file)
    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc, best_file)

    results = {"Last Result": (last_df, last_acc), "Best Result": (best_df, best_acc)}

    # Load poisoned files and sort by poisoning strength if numeric
    poisons = []
    for file in os.listdir(folder):
        if "poisoned" in file and file.endswith(".csv"):
            label = file.replace("results_", "").replace(".csv", "").replace("_", " ").title()
            df, acc = load_csv(os.path.join(folder, file))
            strength = None
            for token in label.split():
                if token.replace("%", "").isdigit():
                    strength = float(token.replace("%", ""))
                    break
            poisons.append((strength if strength is not None else 0, label, df, acc))

    poisons.sort(key=lambda x: x[0])
    for _, label, df, acc in poisons:
        results[label] = (df, acc)

    return results


# -------- Load ROC Data --------
def load_all_roc_data():
    """Scan all model result folders for ROC CSV files."""
    all_roc_curves = []
    for model_name, folder in MODELS.items():
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.startswith("roc_") and file.endswith(".csv"):
                path = os.path.join(folder, file)
                try:
                    df = pd.read_csv(path)
                    if {"fpr", "tpr"}.issubset(df.columns):
                        attack_type = file.replace("roc_", "").replace(".csv", "").replace("_", " ")
                        label = f"{model_name} — {attack_type.title()}"
                        all_roc_curves.append((model_name, attack_type, df, label))
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return all_roc_curves


# -------- Layout --------
app.layout = html.Div([
    html.H1("Universal Detection Model Comparison", style={"textAlign": "center"}),

    dcc.Tabs(
        id="main-tabs",
        value="Models",
        children=[
            dcc.Tab(label="Model Performance", value="Models"),
            dcc.Tab(label="ROC Curves Summary", value="ROC"),
        ],
        style={"marginBottom": "20px"}
    ),

    dcc.Interval(id="interval-refresh", interval=5 * 1000, n_intervals=0),
    html.Div(id="main-container")
])


# -------- Callbacks --------
@app.callback(
    Output("main-container", "children"),
    Input("main-tabs", "value"),
    Input("interval-refresh", "n_intervals")
)
def render_tab(tab, _):
    if tab == "ROC":
        return render_roc_tab()
    else:
        return render_model_tab()


def render_model_tab():
    return html.Div([
        html.H3("Per-Model Detailed Results", style={"textAlign": "center"}),
        dcc.Tabs(
            id="model-tabs",
            value=list(MODELS.keys())[0],
            children=[dcc.Tab(label=m, value=m) for m in MODELS.keys()]
        ),
        html.Div(id="tables-container", style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "gap": "20px"
        })
    ])


@app.callback(
    Output("tables-container", "children"),
    Input("model-tabs", "value"),
    Input("interval-refresh", "n_intervals")
)
def update_tables(selected_model, _):
    folder = MODELS[selected_model]
    results = load_model_results(selected_model, folder)
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
            style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                {"if": {"state": "active"}, "backgroundColor": "#e6f2ff", "border": "1px solid #0074D9"}
            ]
        )
        cards.append(html.Div([
            html.H3(title, style={"textAlign": "center", "marginBottom": "10px"}),
            accuracy_display, table
        ], style={
            "flex": "1",
            "backgroundColor": "white",
            "padding": "10px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            "border": "1px solid #ddd",
            "minWidth": "350px",
            "maxWidth": "45%"
        }))
    return cards


def render_roc_tab():
    all_roc_curves = load_all_roc_data()
    if not all_roc_curves:
        return html.H3("No ROC data found yet. Run your models to generate ROC CSVs.",
                       style={"textAlign": "center", "color": "gray"})

    model_colors = {
        "Logistic Regression": "#1f77b4",
        "Neural Network": "#ff7f0e",
        "Random Forest": "#2ca02c",
        "SVM": "#d62728",
        "XGBoost": "#9467bd",
    }

    traces = []
    for model_name, attack_type, df, label in all_roc_curves:
        color = model_colors.get(model_name, None)
        opacity = 1.0 if "clean" in attack_type.lower() else 0.55
        traces.append(go.Scatter(
            x=df["fpr"], y=df["tpr"],
            mode="lines",
            name=label,
            line=dict(color=color, width=3),
            opacity=opacity
        ))

    fig = go.Figure(traces)
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="gray", dash="dash"))
    fig.update_layout(
        title="Overlayed ROC Curves — All Models & Attack Types",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        height=700,
        legend=dict(x=0.02, y=0.98, bordercolor="lightgray", borderwidth=1),
        plot_bgcolor="#fafafa",
    )

    return html.Div([dcc.Graph(figure=fig)], style={"padding": "20px"})


# -------- Run App --------
if __name__ == "__main__":
    app.run(debug=True)
