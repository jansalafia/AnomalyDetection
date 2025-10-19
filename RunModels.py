from MachineLearning.LogReg.LogisticRegression_ML import run_best_model as run_logreg
from MachineLearning.NeuralNetworks.NeuralNet_ML import run_best_model as run_neuralnet
from MachineLearning.RandomForest.RandomForest_ML import run_best_model as run_randomforest
from MachineLearning.SVM.SVM_ML import run_best_model as run_svm
from MachineLearning.XGBoost.XGBoost_ML import run_best_model as run_xgboost

run_logreg("CSVs/newDataset.csv")
run_neuralnet("CSVs/newDataset.csv")
run_randomforest("CSVs/newDataset.csv")
run_svm("CSVs/newDataset.csv")
run_xgboost("CSVs/newDataset.csv")
