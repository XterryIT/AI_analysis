import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib

# Set print options to suppress scientific notation for readability
np.set_printoptions(suppress=True)

def print_full_evaluation_report(cm):
    """
    Prints a labeled Confusion Matrix followed by a detailed analysis 
    of the Malicious class performance.
    """
    labels = ["Non-DoH", "Benign", "Malicious"]
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    header = f"{'':>15} | {'Pred: ' + labels[0]:>15} | {'Pred: ' + labels[1]:>15} | {'Pred: ' + labels[2]:>15}"
    print(header)
    print("-" * len(header))
    
    for i, row_label in enumerate(labels):
        row_str = f"{'Act: ' + row_label:>15} | {cm[i, 0]:>15} | {cm[i, 1]:>15} | {cm[i, 2]:>15}"
        print(row_str)
    
    print("-" * len(header))
    
    mal_idx = 2
    tp = cm[mal_idx, 2] 
    fn_nondoh = cm[mal_idx, 0] 
    fn_benign = cm[mal_idx, 1] 
    total_fn = fn_nondoh + fn_benign
    
    print("\nMALICIOUS TRAFFIC ANALYSIS (Security Focus):")
    print(f"- True Positives: {tp}")
    print(f"- False Negatives (Missed Attacks): {total_fn}")
    print(f"  --> Breakdown: {fn_nondoh} predicted as Non-DoH + {fn_benign} as Benign.")
    print("="*60 + "\n")

def train_rf_model(X_train, X_test, y_train, y_test):
    """
    Trains the Random Forest model and returns the trained object.
    """
    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n")
    print_full_evaluation_report(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))


    return model 

def main():
    print("#" * 20)
    print("Multiclass Training Pipeline")
    print("#" * 20)

    try:
        df = pd.read_csv('merged_sample.csv')
    except FileNotFoundError:
        print("ERROR: File 'merged_sample.csv' not found!")
        return 

    df.fillna(0, inplace=True)

    X = df.drop(['Label'], axis=1) 
    y = df['Label']

    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)

    print('\nRandom Forest Model Results:')
    

    # 1. Capture the trained model in a variable
    trained_model = train_rf_model(X_train, X_test, y_train, y_test)
    
    print("-"*100)
    print("Saving model and feature list...")

    # 2. Save the TRAINED OBJECT, not the function
    joblib.dump(trained_model, 'rf_model_multiclass.joblib')
    
    # 3. Save feature names to ensure consistency later
    feature_list = list(X.columns)
    joblib.dump(feature_list, 'model_features.joblib')


    print("SUCCESS: Model saved as 'rf_model_multiclass.joblib'")
    print("SUCCESS: Feature list saved as 'model_features.joblib'")

if __name__ == "__main__":
    main()