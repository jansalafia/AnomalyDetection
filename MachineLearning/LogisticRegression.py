import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

#Load the dataset
dataframe = pd.read_csv('OPSAT-AD_modified.csv')


#Data Preperation
y = dataframe['anomaly']
x = dataframe.drop(['anomaly'], axis=1)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)

#Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


#Model Building (Logistic Regression) with increased iterations
LogReg = LogisticRegression(max_iter=100000, random_state=100)
LogReg.fit(x_train, y_train)

#Applying model to make a prediction
y_LogReg_train_pred = LogReg.predict(x_train)
y_LogReg_test_pred = LogReg.predict(x_test)



#Model Evaluation
print("Model Performance:")
print("Training Set Performance:")
print(classification_report(y_train, y_LogReg_train_pred))
print("Test Set Performance:")
print(classification_report(y_test, y_LogReg_test_pred))

# Create visualization plots
plt.figure(figsize=(15, 10))

#Confusion Matrix Heatmap
plt.subplot(2, 1, 1)
cm = confusion_matrix(y_test, y_LogReg_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')


#ROC Curve
plt.subplot(2, 1, 2)
fpr, tpr, _ = roc_curve(y_test, LogReg.predict_proba(x_test)[:,1])
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()

plt.show()

