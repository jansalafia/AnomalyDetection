
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def do_model(path: str, 
             graph: bool = False, 
             show_importance: bool = True):
	"""
	Train Random Forest on `path`, print metrics, and (optionally) show feature importance.

	Feature importance views:
	  - Raw:      signed (can be negative for some RF variants, but usually positive)
	  - Absolute: absolute magnitude (strength of effect)

	Returns:
	  feat_imp (pd.DataFrame) if show_importance else None
	"""
	# 1) Load
	df = pd.read_csv(path)

# 2) Prepare X, y
	X = df.drop(columns=['anomaly','timestamp','channel','label'], errors='ignore')
	y = df['anomaly']
	
	# 3) Stratified 80/20 split
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, test_size=0.2, random_state=100, stratify=y if len(y.unique()) == 2 else None
	)


	# # 3) Standardize (optional for RF, but keeps parity with LR)
	# scaler = StandardScaler()
	# X_scaled = scaler.fit_transform(X)
	# X = pd.DataFrame(X_scaled, columns=X.columns)

	# 4) Split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=100, stratify=y if len(y.unique()) == 2 else None
	)

	# 5) Model
	rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,           	# cap depth
    min_samples_leaf=5,     	# don’t let leaves get too tiny
    random_state=100,
    oob_score=True,         	# extra honest estimate (with bootstrap=True by default)
)
	rf.fit(X_train, y_train)
	# print(rf.feature_importances_)

	# 6) Evaluate
	y_pred_train = rf.predict(X_train)
	y_pred_test  = rf.predict(X_test)

	print("Model Performance:")
	print("Training Set Performance:")
	print(classification_report(y_train, y_pred_train))
	print("Test Set Performance:")
	print(classification_report(y_test, y_pred_test))

	# 7) Optional confusion matrix + ROC
	if graph:
		from sklearn.metrics import ConfusionMatrixDisplay
		disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
		disp.ax_.set_title("Confusion Matrix")
		plt.tight_layout()
		plt.show()

		# ROC Curve (binary)
		if hasattr(rf, "predict_proba"):
			fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
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
		# 8) Feature importances
		importances = rf.feature_importances_
		feat_imp = pd.DataFrame({
			"Feature": X.columns,
			"Importance (Raw)": importances,
		})
		feat_imp["Importance (Abs)"] = feat_imp["Importance (Raw)"].abs()
		feat_imp = feat_imp.sort_values(by="Importance (Abs)", ascending=False).reset_index(drop=True)

		print("\nFeature Importance (Random Forest):")
		print(feat_imp.to_string(index=False))

		# Plot 1: Absolute importance
		plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
		plt.barh(feat_imp["Feature"], feat_imp["Importance (Abs)"])
		plt.gca().invert_yaxis()
		plt.xlabel("Importance (Strength)")
		plt.title("Random Forest — Feature Importance (Absolute)")
		plt.tight_layout()
		plt.show()

	return feat_imp

# Example usage:
do_model("CSVs\OPSAT-AD_modified.csv", graph=False, show_importance=False)

