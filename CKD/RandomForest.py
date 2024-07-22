import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pickle

Train = False

def accuracy(pred, y, showwrong = False):
    classification = pred == y
    correct = sum(classification)
    total = len(classification)

    acc = 100 * (correct / total)
    print("accuracy is:", acc)

    if showwrong == True:
        y_ones = sum(y)
        y_zeros = len(y) - y_ones
        indices = [i for i, x in enumerate(classification) if not x]
        wrong_pred = [x[1] for x in enumerate(pred) if x[0] in indices]
        zeros = sum(wrong_pred)
        ones = len(wrong_pred) - zeros
        print(100 * (zeros/y_zeros), "percent of zeros were incorrectly classified")
        print(100 * (ones/y_ones), "percent of ones were incorrectly classified")
    
    return acc

if __name__ == "__main__":
    # Creating a train-test split

    path = "Chronic_Kidney_Dsease_data.csv"
    df = pd.read_csv(path)

    # Note we oversample the zero class due to it's significant underrepresentation

    oversample = RandomOverSampler(sampling_strategy=1)

    features = ["MuscleCramps", "Itching", "SleepQuality", "DietQuality", "GFR", "SerumCreatinine", "ProteinInUrine", "FastingBloodSugar", "CholesterolTriglycerides", "MedicationAdherence", "FamilyHistoryDiabetes", "SystolicBP", "DiastolicBP", "ACR"]

    X = df[features]
    y = df['Diagnosis']

    X, y = oversample.fit_resample(X, y)

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train model
    if Train == True:
        model = RandomForestClassifier(max_depth=12, random_state=0)
        model.fit(X_train, y_train)
    else:
        filename = 'RF_model.sav'
        model = pickle.load(open(filename, 'rb'))

    # Save model
    filename = 'RF_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # Train and Test accuracies
    pred = model.predict(X_train)
    accuracy(pred, y_train)

    pred = model.predict(X_test)
    accuracy(pred, y_test)

    # 5-fold cv
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 1)

    accuracies_train = []
    accuracies_test = []

    for train, test in kfold.split(X):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        model.fit(X_train, y_train)

        # Train and Test accuracies
        pred = model.predict(X_train)
        accuracies_train.append(accuracy(pred, y_train))

        pred = model.predict(X_test)
        accuracies_test.append(accuracy(pred, y_test))


    folds = np.array([1,2,3,4,5])
    dat = np.c_[folds, np.array(accuracies_train), np.array(accuracies_test)]
    df = pd.DataFrame(dat, columns=['Fold', 'Training Accuracies', 'Testing Accuracies'])
    df.to_csv('5foldcv.csv', index = False) 

