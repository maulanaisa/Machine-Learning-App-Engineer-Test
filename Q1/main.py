import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

path = r'src\Q1.csv'
test_size = 0.3     # Testing data size portion
n_estimator = 10

def main():
    df = pd.read_csv(path)
    df.drop(['Timestamp'],inplace=True, axis=1)

    # Split Data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

    # Data Normalization
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Random Over-Sample for Imbalaced Training Data
    overSampling = RandomOverSampler(random_state=42)
    X_trainOver, y_trainOver = overSampling.fit_resample(X_train, y_train)

    # Random Forest Model
    classifier = RandomForestClassifier(n_estimators = n_estimator, criterion = 'entropy', random_state = 0)
    classifier.fit(X_trainOver, y_trainOver)

    # Metrics
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()