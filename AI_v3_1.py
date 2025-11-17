import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# Here i Will try to decrease FN in Malicious detections by using: class_weight, AdaBoost, Oversampling.
# Data we using as same for AI_v3

np.set_printoptions(suppress=True)


def Random_Forest_model(X_train, X_test, y_train, y_test, feature_names):

    weights = {0: 1, 1: 1, 2: 8}

    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    # 100 trees is a good starting point
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Explicitly set labels to ensure [0, 1, 2] order in the matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix (3x3):")
    print(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    # Use target_names to make the report readable
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))
    

#AdaBost Model
def AdaBoost_Model(X_train, X_test, y_train, y_test, feature_names):

    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    # AdaBoost often uses simple Decision Trees as its base
    base_estimator = DecisionTreeClassifier(max_depth=2) 

    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100, # 100 specialists
        random_state=42
    )


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Explicitly set labels to ensure [0, 1, 2] order in the matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix (3x3):")
    print(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    # Use target_names to make the report readable
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))
    

#
# --- Main function ---
#
def main():
    
    print("#" * 20)
    print("Multiclass")
    print("#" * 20)

    try:
        df = pd.read_csv('merged_sample.csv')
    except FileNotFoundError:
        print("ERROR: File 'merged_sample.csv' not found!")
        return 

    # Fill any missing values (NaN) with 0
    df.fillna(0, inplace=True)


    X = df.drop(['Label'], axis=1) 

    y = df['Label']

    # Print the class distribution
    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)


    print('\nRandom Forest Model Results:')
    Random_Forest_model(X_train, X_test, y_train, y_test, X.columns)
    print("-"*100)
    


if __name__ == "__main__":
    main()




# Condition 1:
#
# We are implmented a punishment for wrong choice for model especialy for Malicious DoH (class 2)
#
#
# Conclusion 1:
# 
# We see that presents result are worse than previous and we don`t see any increases`
# Especialy here methods of weights does not working
#
#
#Previous Results:
#--- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.12%
# Confusion Matrix (3x3):
# [[ 5837   116    12]
#  [   72 11811    83]
#  [   29   138  5845]]
#
#
#
#
#Present Results:
# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.10%
# Confusion Matrix (3x3):
# [[ 5840   115    10]
#  [   70 11813    83]
#  [   27   151  5834]]

# Classification Report (0=NonDoH, 1=Benign, 2=Malicious):
#                precision    recall  f1-score   support

#    NonDoH (0)       0.98      0.98      0.98      5965
#    Benign (1)       0.98      0.99      0.98     11966
# Malicious (2)       0.98      0.97      0.98      6012

#      accuracy                           0.98     23943
#     macro avg       0.98      0.98      0.98     23943
#  weighted avg       0.98      0.98      0.98     23943
#
#
#
#
# Condition 2:
#
# We use AdaBoost with Random Forest as base estimator
#
#