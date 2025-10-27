import os
import re
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import pandas as pd

# Initialize Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Universal Model Comparison Dashboard"

# Keep your model -> folder mapping exactly as you had it
MODELS = {
    "Logistic Regression": "Results/LogRegResults",
    "Neural Network": "Results/NeuralNetworksResults",
    "Random Forest": "Results/RandomForestResults",
    "SVM": "Results/SVMResults",
    "XGBoost": "Results/XGBoostResults",
}

# IMPORTANT: filename base for each model (fixes Last/Best not showing)
# e.g. Logistic Regression uses results_logreg.csv (not results_logisticregression.csv)
FILENAME_BASES = {
    "Logistic Regression": "logreg",
    "Neural Network": "neuralnet",
    "Random Forest": "randomforest",
    "SVM": "svm",
    "XGBoost": "xgboost",
}


# ----------- Helpers -----------
def load_csv(path):
    """Load CSV safely and try to extract accuracy from a row labelled 'accuracy'."""
    if os.path.exists(path):
        df = pd.read_csv(path).round(3)
        acc = None
        # protect if df empty or malformed
        try:
            first_col = df.columns[0]
            if "accuracy" in df[first_col].astype(str).values:
                row = df[df[first_col] == "accuracy"]
                # try find accuracy in precision column
                if "precision" in row.columns and len(row) > 0:
                    acc = float(row["precision"].values[0])
                df = df[df[first_col] != "accuracy"]
        except Exception:
            pass
        return df.reset_index(drop=True), acc
    else:
        # consistent placeholder for missing file
        return pd.DataFrame([{"status": "No results found"}]), None


def _is_valid_df(df):
    """Return True if df looks like real results (not the 'No results found' placeholder)."""
    return not df.empty and "status" not in df.columns


def update_best(last_df, last_acc, best_df, best_acc, best_path):
    """
    Update best run by comparing f1-score per row and saving back to best_path.
    Handles missing/bad best_df gracefully.
    """
    # if last isn't valid, nothing to update
    if not _is_valid_df(last_df):
        return best_df, best_acc

    # if best missing or not containing f1-score -> take last as best
    if not _is_valid_df(best_df) or "f1-score" not in best_df.columns:
        full_best = last_df.copy()
        best_acc = last_acc
    else:
        full_best = best_df.copy()
        # compare row-by-row for f1-score (only for overlapping indices)
        for idx in last_df.index:
            if idx < len(full_best):
                try:
                    if float(last_df.loc[idx, "f1-score"]) > float(full_best.loc[idx, "f1-score"]):
                        full_best.loc[idx] = last_df.loc[idx]
                except Exception:
                    # if something goes wrong (missing columns or values), ignore
                    pass

    # reattach accuracy row at top if available
    if best_acc is not None and _is_valid_df(full_best):
        # create accuracy row with same columns
        cols = list(full_best.columns)
        acc_row = {cols[0]: "accuracy", "precision": best_acc}
        for c in cols[2:]:
            acc_row[c] = ""
        acc_df = pd.DataFrame([acc_row])
        full_best = pd.concat([acc_df, full_best], ignore_index=True)

    # save best even when not present previously (if there is something to save)
    try:
        if _is_valid_df(full_best):
            full_best.to_csv(best_path, index=False)
    except Exception:
        pass

    return full_best, best_acc


def parse_poisoned_filename(filename):
    """
    Parse a poisoned filename to extract (attack_name, strength_label, strength_numeric_or_None).
    Accepts many patterns:
      - results_model_poisoned_2.csv -> attack_name='Label Flip' (default), strength='2', numeric 2.0
      - results_model_poisoned_labelflip_2.csv -> attack_name='Labelflip', strength='2', numeric 2.0
      - results_model_poisoned_label_flip_10pct.csv -> attack_name='Label Flip', strength='10pct', numeric 10.0 (if endswith 'pct' or '%')
      - results_model_poisoned_gaussian_noise.csv -> attack_name='Gaussian Noise', strength='Unknown'
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    # find 'poisoned' index
    try:
        idx = parts.index("poisoned")
    except ValueError:
        return ("Poisoned", "Unknown", None)

    tail = parts[idx + 1 :]  # tokens after 'poisoned'
    if not tail:
        return ("Poisoned", "Unknown", None)

    # try interpret last token as strength if it contains digits or ends with pct/% etc.
    last = tail[-1]
    strength_numeric = None
    strength_label = last

    # clean token (remove 'pct' suffix and '%' if present)
    m_pct = re.match(r"^(\d+(?:\.\d+)?)(?:pct|%?)$", last)
    if m_pct:
        try:
            strength_numeric = float(m_pct.group(1))
            strength_label = str(m_pct.group(1))
        except Exception:
            strength_numeric = None
    else:
        # check if purely numeric
        if re.match(r"^\d+(\.\d+)?$", last):
            strength_numeric = float(last)
            strength_label = last
        else:
            # maybe last token has digits within (like '10pct' or '2percent')
            m = re.search(r"(\d+(?:\.\d+)?)", last)
            if m:
                try:
                    strength_numeric = float(m.group(1))
                    strength_label = m.group(1)
                except Exception:
                    strength_numeric = None

    # attack name is tail without last token if last token was detected as strength
    attack_tokens = tail
    if strength_numeric is not None:
        attack_tokens = tail[:-1]

    if not attack_tokens:
        # default attack name if none provided
        attack_name = "Label Flip"
    else:
        attack_name = " ".join(attack_tokens).replace("-", " ").replace("_", " ").title()

    return (attack_name, strength_label, strength_numeric)


def load_model_results(model_name, folder):
    """
    Returns:
      - base_results: dict for Last Result and Best Result -> (df, acc)
      - attacks: dict attack_name -> list of entries for that attack
           each entry: dict { 'label': filename_label, 'strength_label':..., 'strength_numeric':..., 'df':..., 'acc':..., 'file':... }
    """
    base_results = {}
    attacks = {}

    base = FILENAME_BASES.get(model_name, model_name.lower().replace(" ", ""))
    last_file = os.path.join(folder, f"results_{base}.csv")
    best_file = os.path.join(folder, f"results_{base}_best.csv")

    last_df, last_acc = load_csv(last_file)
    best_df, best_acc = load_csv(best_file)

    # update best and persist
    best_df, best_acc = update_best(last_df, last_acc, best_df, best_acc, best_file)

    base_results["Last Result"] = (last_df, last_acc)
    base_results["Best Result"] = (best_df, best_acc)

    # iterate files and collect poisoned files
    if os.path.isdir(folder):
        for fname in sorted(os.listdir(folder)):
            if "poisoned" in fname.lower() and fname.lower().endswith(".csv"):
                attack_name, strength_label, strength_numeric = parse_poisoned_filename(fname.lower())
                df, acc = load_csv(os.path.join(folder, fname))
                entry = {
                    "label": fname.replace(".csv", ""),
                    "strength_label": strength_label,
                    "strength_numeric": strength_numeric,
                    "df": df,
                    "acc": acc,
                    "file": os.path.join(folder, fname),
                }
                attacks.setdefault(attack_name, []).append(entry)

    # sort entries in each attack by numeric strength ascending, then by label
    for attack_name, entries in attacks.items():
        entries_sorted = sorted(entries, key=lambda e: (e["strength_numeric"] is None, e["strength_numeric"] if e["strength_numeric"] is not None else float("inf"), str(e["strength_label"])))
        attacks[attack_name] = entries_sorted

    return base_results, attacks


# ----------- Layout -----------
app.layout = html.Div([
    html.H1("Universal Detection Model Comparison", style={"textAlign": "center"}),

    dcc.Tabs(
        id="model-tabs",
        value=list(MODELS.keys())[0],
        children=[dcc.Tab(label=m, value=m) for m in MODELS.keys()],
        style={"marginBottom": "10px"}
    ),

    html.Div([
        html.Div([
            html.Label("Select Attack (group):"),
            dcc.Dropdown(id="attack-dropdown", options=[], value="All Attacks", clearable=False, style={"width": "320px"})
        ], style={"display": "inline-block", "verticalAlign": "middle", "marginLeft": "20px"}),
        dcc.Interval(id="interval-refresh", interval=5 * 1000, n_intervals=0)
    ], style={"display": "flex", "alignItems": "center", "gap": "20px", "marginBottom": "10px"}),

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
    Output("attack-dropdown", "options"),
    Output("attack-dropdown", "value"),
    Input("interval-refresh", "n_intervals"),
    Input("model-tabs", "value"),
    Input("attack-dropdown", "value")
)
def update_dashboard(n_intervals, selected_model, selected_attack):
    """
    - Refreshes every interval and whenever model or attack selection changes.
    - Returns: (cards, dropdown_options, dropdown_value_to_set)
    """
    folder = MODELS.get(selected_model)
    if folder is None:
        return [html.Div("Model folder not found")], [], "All Attacks"

    base_results, attacks = load_model_results(selected_model, folder)

    # build dropdown options: All Attacks + each detected attack
    attack_names = sorted(attacks.keys())
    dd_options = [{"label": "All Attacks", "value": "All Attacks"}] + [{"label": a, "value": a} for a in attack_names]

    # if selected_attack is None or not present, set default to All Attacks
    if selected_attack not in [o["value"] for o in dd_options]:
        selected_attack = "All Attacks"

    cards = []

    # Always show Last & Best results first (if valid)
    for base_title in ["Last Result", "Best Result"]:
        df, acc = base_results.get(base_title, (pd.DataFrame([{"status": "No results found"}]), None))
        display_df = df[df.iloc[:, 0] != "accuracy"].reset_index(drop=True) if _is_valid_df(df) else df

        accuracy_display = (html.H2(f"Accuracy: {acc:.3f}", style={"color": "#0074D9", "textAlign": "center"})
                            if acc is not None else html.H2("Accuracy: —", style={"color": "gray", "textAlign": "center"}))

        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in display_df.columns] if _is_valid_df(display_df) else [{"name": "status", "id": "status"}],
            data=display_df.to_dict("records"),
            style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}]
        )

        card = html.Div([
            html.H3(base_title, style={"textAlign": "center", "marginBottom": "6px"}),
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

    # Helper to produce card for a poisoned entry
    def make_poison_card(entry, attack_label):
        df = entry["df"]
        acc = entry["acc"]
        display_df = df[df.iloc[:, 0] != "accuracy"].reset_index(drop=True) if _is_valid_df(df) else df

        accuracy_display = (html.H4(f"Accuracy: {acc:.3f}", style={"color": "#0074D9", "textAlign": "center"})
                            if acc is not None else html.H4("Accuracy: —", style={"color": "gray", "textAlign": "center"}))

        title = f"{attack_label} — {entry['strength_label']}" if entry["strength_label"] else attack_label

        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in display_df.columns] if _is_valid_df(display_df) else [{"name": "status", "id": "status"}],
            data=display_df.to_dict("records"),
            style_table={"overflowX": "auto", "maxHeight": "380px", "overflowY": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#f8f8f8", "fontWeight": "bold"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}]
        )

        card = html.Div([
            html.H4(title, style={"textAlign": "center", "marginBottom": "6px"}),
            accuracy_display,
            table
        ], style={
            "flex": "1",
            "backgroundColor": "white",
            "padding": "8px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
            "border": "1px solid #e6e6e6",
            "minWidth": "300px",
            "maxWidth": "30%"
        })
        return card

    # Show attacks according to selected_attack dropdown
    if selected_attack == "All Attacks":
        # group attacks in alphabetical order; each attack's entries are already sorted by strength
        for attack_name in attack_names:
            # an attack heading card (spanning full width)
            cards.append(html.Div(html.H3(attack_name, style={"textAlign": "center", "width": "100%", "marginTop": "8px"})))
            for entry in attacks[attack_name]:
                cards.append(make_poison_card(entry, attack_name))
    else:
        # show only the selected attack's entries
        if selected_attack in attacks:
            for entry in attacks[selected_attack]:
                cards.append(make_poison_card(entry, selected_attack))
        else:
            # no attack files found
            cards.append(html.Div(f"No poisoned files found for attack '{selected_attack}'", style={"padding": "10px"}))

    return cards, dd_options, selected_attack


# ----------- Run -----------
if __name__ == "__main__":
    app.run(debug=True)
