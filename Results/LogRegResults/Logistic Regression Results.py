# app_compare_autoupdate.py
import dash
from dash import html, dash_table, dcc
from dash.dependencies import Output, Input
import pandas as pd
import os

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Anomaly Logistic Regression Detection Comparison"

# Paths to CSVs
LAST_RESULT_CSV = "Results/LogRegResults/results_logreg.csv"
BEST_RESULT_CSV = "Results/LogRegResults/results_logreg_best.csv"
POISONED_RESULT_CSV = "Results/LogRegResults/results_logreg_poisoned.csv"

def load_csv(file_path):
    """Safely load CSV and round numeric columns to 3 decimal places"""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.reset_index(inplace=True, drop=True)
        # Round numeric columns to 3 decimals
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].round(5)
        return df
    else:
        return pd.DataFrame({"message": [f"{file_path} not found"]})

def update_best_result(last_df, best_df, output_path=BEST_RESULT_CSV):
    """Update best_df with last_df if last_df has better f1-score"""
    if best_df.empty or "f1-score" not in best_df.columns:
        last_df.to_csv(output_path, index=False)
        return last_df

    updated = best_df.copy()
    for idx in last_df.index:
        if idx in best_df.index:
            if last_df.loc[idx, "f1-score"] > best_df.loc[idx, "f1-score"]:
                updated.loc[idx] = last_df.loc[idx]
    updated.to_csv(output_path, index=False)
    return updated

# App layout
app.layout = html.Div([
    html.H1("Anomaly Logistic Regression Detection Results Comparison"),

    # Interval for auto-refresh
    dcc.Interval(id='interval-refresh', interval=5000, n_intervals=0),  # every 5 seconds

    html.Div(id="tables-container", style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'})
])

# Callback to refresh tables
@app.callback(
    Output("tables-container", "children"),
    Input("interval-refresh", "n_intervals")
)
def refresh_tables(n):
    # Load latest data
    last_result = load_csv(LAST_RESULT_CSV)
    best_result = load_csv(BEST_RESULT_CSV)
    poisoned_result = load_csv(POISONED_RESULT_CSV)

    # Update best result
    best_result = update_best_result(last_result, best_result)

    # Return side-by-side tables
    table_style = {'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'}
    cell_style = {'textAlign': 'left', 'padding': '5px'}
    header_style = {'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}

    def make_table(df, title):
        return html.Div([
            html.H3(title),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict("records"),
                style_table=table_style,
                style_cell=cell_style,
                style_header=header_style
            )
        ], style={'flex': '1'})

    return [
        make_table(last_result, "Last Result"),
        make_table(best_result, "Best Result"),
        make_table(poisoned_result, "Poisoned Data Result")
    ]

if __name__ == "__main__":
    app.run(debug=True)
