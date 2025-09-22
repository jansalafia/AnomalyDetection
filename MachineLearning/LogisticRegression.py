# LogisticRegression_feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def do_model(path: str,
            graph: bool = False,
            show_importance: bool = True,):
    """
    Train Logistic Regression on `path`, print metrics, and (optionally) show feature importance.

    Feature importance views:
      - Coefficient (raw): keeps sign (direction of effect on anomaly odds)
      - Importance (abs):  absolute magnitude (strength of effect, regardless of sign)

    Returns:
      feat_imp (pd.DataFrame) if show_importance else None
    """
    
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')

    # # 3) Standardize (so coefficients are comparable across features)         //STANDARDIZING ALTERS FEATURE IMPORTANCE
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=100, stratify=y
    )

    # 5) Model
    logreg = LogisticRegression(max_iter=100_000, random_state=100)
    logreg.fit(X_train, y_train)

    # 6) Evaluate
    y_pred_train = logreg.predict(X_train)
    y_pred_test  = logreg.predict(X_test)

    print("Model Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred_test))

    # 7) Optional confusion matrix + ROC
    if graph:
        # Confusion Matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        disp.ax_.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (binary)
        if hasattr(logreg, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
            auc_val = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()

    feat_imp = None
    
    if show_importance:
        # 8) Coefficients → Feature importance
        # For binary logistic regression, coef_ has shape (1, n_features)
        coefs = logreg.coef_[0]
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient (Raw)": coefs,
        })
        feat_imp["Importance (Abs)"] = feat_imp["Coefficient (Raw)"].abs()

        # sort by absolute importance (strength), keep sign visible alongside
        feat_imp = feat_imp.sort_values(by="Importance (Abs)", ascending=False).reset_index(drop=True)

        # Print a compact table
        print("\nFeature Importance (Logistic Regression):")
        print(feat_imp.to_string(index=False))

        # Plot 1: Absolute importance (strength)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (Abs)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Absolute Coefficient (Strength)")
        plt.title("Logistic Regression — Feature Importance (Absolute)")
        plt.tight_layout()
        plt.show()

        # Plot 2: Signed coefficients (direction)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Coefficient (Raw)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient (Signed)")
        plt.title("Logistic Regression — Feature Coefficients (Direction)")
        plt.tight_layout()
        plt.show()

    return feat_imp

# === Examples (match your original usage) ===
do_model('CSVs/OPSAT-AD_modified.csv', graph=False, show_importance=True)
# do_model('CSVs/Output/merged_OPSSAT_segments.csv', graph=True, show_importance=True)
