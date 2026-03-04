import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# ---------------- Initialize Dash ----------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Universal Model Comparison Dashboard"

# ---------------- Models ----------------
MODELS = {
    "Logistic Regression": "Results/LogRegResults",
    "Neural Network": "Results/NeuralNetworksResults",
    "Random Forest": "Results/RandomForestResults",
    "SVM": "Results/SVMResults",
    "XGBoost": "Results/XGBoostResults",
}

MODEL_FILE_MAP = {
    "Logistic Regression": ("results_logreg.csv", "results_logreg_best.csv"),
    "Neural Network": ("results_neuralnet.csv", "results_neuralnet_best.csv"),
    "Random Forest": ("results_randomforest.csv", "results_randomforest_best.csv"),
    "SVM": ("results_svm.csv", "results_svm_best.csv"),
    "XGBoost": ("results_xgboost.csv", "results_xgboost_best.csv"),
}

# ---------------- Helper Functions ----------------
def load_csv(path):
    """Load CSV and extract accuracy row if exists."""
    if not os.path.exists(path):
        return pd.DataFrame([{"status": "No results found"}]), None
    try:
        df = pd.read_csv(path).round(3)
    except Exception as e:
        return pd.DataFrame([{"error": f"Could not read CSV: {e}"}]), None

    acc = None
    if not df.empty and "accuracy" in df.iloc[:, 0].values:
        row = df[df.iloc[:, 0] == "accuracy"]
        if not row.empty:
            try:
                acc = float(row["precision"].values[0])
            except:
                pass
        df = df[df.iloc[:, 0] != "accuracy"]

    return df.reset_index(drop=True), acc

def update_best(last_df, last_acc, best_df, best_acc, best_path):
    """Update best results based on F1-score."""
    if best_df.empty or "f1-score" not in best_df.columns:
        full_best = last_df.copy()
        best_acc = last_acc
    else:
        full_best = best_df.copy()
        for idx in last_df.index:
            if idx < len(best_df) and pd.notna(last_df.loc[idx, "f1-score"]) and last_df.loc[idx, "f1-score"] > best_df.loc[idx, "f1-score"]:
                full_best.loc[idx] = last_df.loc[idx]

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
    """Load last/best/poisoned results for a model."""
    if model_name not in MODEL_FILE_MAP:
        return {"Error": (pd.DataFrame([{"status": "Unknown model"}]), None)}

    last_file, best_file = MODEL_FILE_MAP[model_name]
    last_df, last_acc = load_csv(os.path.join(folder, last_file))
    best_df, best_acc = load_csv(os.path.join(folder, best_file))
    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc, os.path.join(folder, best_file))

    results = {"Last Result": (last_df, last_acc), "Best Result": (best_df, best_acc)}

    # Load poisoned files
    for file in os.listdir(folder):
        if "poisoned" in file and file.endswith(".csv"):
            label = file.replace("results_", "").replace(".csv", "").replace("_", " ").title()
            df, acc = load_csv(os.path.join(folder, file))
            results[label] = (df, acc)
    return results

def load_all_roc_data():
    """Load ROC CSVs with fpr/tpr."""
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
                        attack_type = file.replace("roc_", "").replace(".csv", "").replace("_", " ").title()
                        all_roc_curves.append((model_name, attack_type, df))
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return all_roc_curves

def compute_auc(df):
    """Compute AUC from fpr/tpr using trapezoidal rule."""
    fpr = df["fpr"].values
    tpr = df["tpr"].values
    auc_val = 0.0
    for i in range(1, len(fpr)):
        auc_val += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return round(auc_val, 3)

def create_confusion_matrix(df_preds, y_true_col="y_true", y_pred_col="y_pred"):
    """Generate confusion matrix heatmap figure if y_true/y_pred exist."""
    if y_true_col not in df_preds.columns or y_pred_col not in df_preds.columns:
        return html.Div("No confusion matrix data available.", style={"textAlign": "center", "color": "gray"})

    cm = confusion_matrix(df_preds[y_true_col], df_preds[y_pred_col])
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    return dcc.Graph(figure=fig)

# ---------------- Layout ----------------
app.layout = html.Div([
    html.H1("Universal Detection Model Comparison", style={"textAlign": "center"}),
    dcc.Tabs(id="main-tabs", value="Models", children=[
        dcc.Tab(label="Model Performance", value="Models"),
        dcc.Tab(label="ROC Curves Summary", value="ROC"),
    ], style={"marginBottom": "20px"}),

    dcc.Interval(id="interval-refresh", interval=60000, n_intervals=0),
    html.Div(id="main-container")
])

# ---------------- Callbacks ----------------
@app.callback(
    Output("main-container", "children"),
    Input("main-tabs", "value"),
    Input("interval-refresh", "n_intervals")
)
def render_tab(tab, _):
    if tab == "ROC":
        return render_roc_tab()
    return render_model_tab()

def render_model_tab():
    return html.Div([
        html.H3("Per-Model Detailed Results", style={"textAlign": "center"}),
        dcc.Tabs(id="model-tabs", value=list(MODELS.keys())[0],
                 children=[dcc.Tab(label=m, value=m) for m in MODELS.keys()]),
        html.Div(id="tables-container", style={"display":"flex", "flexWrap":"wrap", "justifyContent":"center", "gap":"20px"})
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
        display_df = df[df.iloc[:,0] != "accuracy"].reset_index(drop=True)
        auc_val = None
        if {"fpr","tpr"}.issubset(df.columns):
            auc_val = compute_auc(df)
        accuracy_display = html.H2(
            f"Accuracy: {acc:.3f}" if acc is not None else "Accuracy: —",
            style={"color": "#0074D9" if acc is not None else "gray", "textAlign": "center"}
        )
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in display_df.columns],
            data=display_df.to_dict("records"),
            style_table={"overflowX":"auto","maxHeight":"400px","overflowY":"auto"},
            style_cell={"textAlign":"center","padding":"6px"},
            style_header={"backgroundColor":"#f0f0f0","fontWeight":"bold"},
            style_data_conditional=[
                {"if":{"row_index":"odd"}, "backgroundColor":"#fafafa"},
                {"if":{"state":"active"}, "backgroundColor":"#e6f2ff","border":"1px solid #0074D9"}
            ]
        )
        auc_display = html.H4(f"AUC: {auc_val}" if auc_val is not None else "AUC: —", style={"textAlign":"center"})
        # Placeholder for confusion matrix (needs separate CSV with y_true/y_pred)
        cm_graph = html.Div("Confusion matrix not available", style={"textAlign":"center","color":"gray"})

        cards.append(html.Div([html.H3(title, style={"textAlign":"center"}), accuracy_display, auc_display, table, cm_graph],
            style={"flex":"1","backgroundColor":"white","padding":"10px","borderRadius":"10px","boxShadow":"0 2px 8px rgba(0,0,0,0.1)",
                   "border":"1px solid #ddd","minWidth":"350px","maxWidth":"45%"}))
    return cards

def render_roc_tab():
    all_roc_curves = load_all_roc_data()
    if not all_roc_curves:
        return html.H3("No ROC data found yet. Run your models to generate ROC CSVs.", style={"textAlign":"center","color":"gray"})

    model_colors = {
        "Logistic Regression": "#1f77b4",
        "Neural Network": "#ff7f0e",
        "Random Forest": "#2ca02c",
        "SVM": "#d62728",
        "XGBoost": "#9467bd",
    }

    fig = go.Figure()
    for model_name, attack_type, df in all_roc_curves:
        color = model_colors.get(model_name)
        auc_val = compute_auc(df)
        fig.add_trace(go.Scatter(
            x=df["fpr"], y=df["tpr"],
            mode="lines",
            name=f"{model_name} {attack_type} | AUC={auc_val}",
            line=dict(color=color, width=3)
        ))

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    fig.update_layout(
        title="Overlayed ROC Curves — All Models & Attack Types",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        height=700,
        legend=dict(x=1, y=0, xanchor="right", yanchor="bottom", bordercolor="lightgray", borderwidth=1),
        plot_bgcolor="#fafafa",
    )
    return html.Div([dcc.Graph(figure=fig)], style={"padding":"20px"})

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
