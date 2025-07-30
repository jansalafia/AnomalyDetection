import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

#Start Timer
start_time = time.perf_counter()

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
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_LogReg_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

#Feature Importance Plot
plt.subplot(2, 2, 2)
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': abs(LogReg.coef_[0])
})
importance = importance.sort_values('Importance', ascending=False)
sns.barplot(data=importance.head(10), x='Importance', y='Feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Absolute Coefficient Value')

#Distribution of Predictions
plt.subplot(2, 2, 3)
sns.histplot(data=pd.DataFrame({
    'True': y_test,
    'Predicted': y_LogReg_test_pred
}).melt(), x='value', hue='variable')
plt.title('Distribution of True vs Predicted Values')

#ROC Curve
plt.subplot(2, 2, 4)
fpr, tpr, _ = roc_curve(y_test, LogReg.predict_proba(x_test)[:,1])
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()


# Timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Task executed in: {elapsed_time:.5f} seconds")

plt.show()

