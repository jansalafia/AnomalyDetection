import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import IsolationForest

#Load the dataset
dataframe = pd.read_csv('CSVs\OPSAT-AD_modified.csv')


#Data Preperation
y = dataframe['anomaly']
x = dataframe.drop(['anomaly'], axis=1)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)

#Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


#Model Building (Isolatio Forest)
IsoForest = IsolationForest(random_state=100)
IsoForest.fit(x_train, y_train)

#Applying model to make a prediction
y_IsoForest_train_pred = IsoForest.predict(x_train)
y_IsoForest_test_pred = IsoForest.predict(x_test)



#Model Evaluation
print("Model Performance:")
print("Training Set Performance:")
print(classification_report(y_train, y_IsoForest_train_pred, zero_division=1))
print("Test Set Performance:")
print(classification_report(y_test, y_IsoForest_test_pred, zero_division=1))

# Create visualization plots
plt.figure(figsize=(15, 10))

#Confusion Matrix Heatmap
plt.subplot(2, 1, 1)
cm = confusion_matrix(y_test, y_IsoForest_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')


#ROC Curve
plt.subplot(2, 1, 2)
fpr, tpr, _ = roc_curve(y_test, IsoForest.predict_proba(x_test)[:,1])
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()

plt.show()

