import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Set print options to suppress scientific notation for readability
np.set_printoptions(suppress=True)

def print_full_evaluation_report(cm):
    """
    Prints a labeled Confusion Matrix followed by a detailed analysis 
    of the Malicious class performance.
    
    Assumed class order: [0: Non-DoH, 1: Benign, 2: Malicious]
    """
    # Define class labels for the report
    labels = ["Non-DoH", "Benign", "Malicious"]
    
    # 1. PRINT LABELED MATRIX
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    # Print Column Headers (Predictions)
    # The formatting {:>15} aligns text to the right with 15 spaces width
    header = f"{'':>15} | {'Pred: ' + labels[0]:>15} | {'Pred: ' + labels[1]:>15} | {'Pred: ' + labels[2]:>15}"
    print(header)
    print("-" * len(header))
    
    # Print Rows (Actual / True Labels)
    for i, row_label in enumerate(labels):
        row_str = f"{'Act: ' + row_label:>15} | {cm[i, 0]:>15} | {cm[i, 1]:>15} | {cm[i, 2]:>15}"
        print(row_str)
    
    print("-" * len(header))
    
    # 2. PRINT MALICIOUS ANALYSIS (The logic we added earlier)
    # Index 2 represents the Malicious class
    mal_idx = 2
    
    # Retrieve values from the matrix
    tp = cm[mal_idx, 2] # True Positives (Malicious correctly identified)
    fn_nondoh = cm[mal_idx, 0] # Missed: Malicious -> Non-DoH
    fn_benign = cm[mal_idx, 1] # Missed: Malicious -> Benign (Critical)
    total_fn = fn_nondoh + fn_benign
    
    print("\nMALICIOUS TRAFFIC ANALYSIS (Security Focus):")
    print(f"- True Positives: {tp}")
    print(f"- False Negatives (Missed Attacks): {total_fn}")
    print(f"  --> Breakdown: {fn_nondoh} predicted as Non-DoH + {fn_benign} as Benign.")
    print("="*60 + "\n")

def random_forest_model(x_train, x_test, y_train, y_test, feature_names):
    """
    feature_names - it is columns in data (x.columns)
    """
    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    # 100 trees is a good starting point
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Explicitly set labels to ensure [0, 1, 2] order in the matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n")
    print_full_evaluation_report(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    # Use target_names to make the report readable
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))
    
    #
    # --- 🚀 NEW BLOCK: FEATURE IMPORTANCE 🚀 ---
    #
    print("\n--- FEATURE IMPORTANCE ---")
    
    # 1. Get the importances from the trained model
    importances = model.feature_importances_
    
    # 2. Create a DataFrame for a nice display
    # (Combine feature names with their importance scores)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 3. Sort by importance (from most important to least important)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # 4. Print the Top 10 most important features
    print("Top 10 features used by the model:")
    print(feature_importance_df.head(10))
    # -----------------------------------------------


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

    # X = Features (all columns EXCEPT the one we are predicting)
    x = df.drop(['Label'], axis=1)
    # y = Target (the ONE column we are predicting)
    y = df['Label']

    # Print the class distribution
    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    # Split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)

    # --- CHANGE IS HERE ---
    # We pass x.columns (the column names) to the function
    print('\nRandom Forest Model Results:')
    random_forest_model(x_train, x_test, y_train, y_test, x.columns)
    print("-"*100)
    
    # (I commented out DecisionTree to speed up the test)
    # print('Decision Tree Model Results:')
    # decision_tree_model(x_train, x_test, y_train, y_test, x.columns)
    # print("-"*100)


if __name__ == "__main__":
    main()






# Condition 1:  
# We are looking what a features are most important for Random Forest model in multiclass system (NonDoH, Benign, Malicious).
#
# Conclusion 1:
# In test we use system with 3 params to 1 model (Multiclass: NonDoH, Benign, Malicious).
# 
# Random Forest it is feature selection, and we can see that the most important parameters are:
#
# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 99.73%
# Confusion Matrix (3x3):
# [[2989   20    0]
#  [   4 5985    0]
#  [   1    7 2994]]
#
#
# Top 10 features used by the model:
#                                Feature  Importance
# 11                    PacketLengthMode    0.136571
# 9                     PacketLengthMean    0.122147
# 7                 PacketLengthVariance    0.093094
# 10                  PacketLengthMedian    0.087382
# 8        PacketLengthStandardDeviation    0.068296
# 14  PacketLengthCoefficientofVariation    0.065780
# 3                        FlowBytesSent    0.063094
# 5                    FlowBytesReceived    0.057504
# 0                           SourcePort    0.051513
# 2                             Duration    0.025594
# ----------------------------------------------------------------------------------------------------





# Condition 2:
# If we delete all the calculated features (like mean, median, mode, stddev, variance, covariation) 
# and leave only basic features (SourcePort,DestinationPort,Duration,FlowBytesSent,FlowSentRate,FlowBytesReceived,FlowReceivedRate,Label)
#
#
# Conclusion 2:
#
#--- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.12%
# Confusion Matrix (3x3):
# [[ 5837   116    12]
#  [   72 11811    83]
#  [   29   138  5845]]

#   5837 - TP for NonDoH
# False Negatives (Missed Attacks): 167 (29 predicted as Non-DoH + 138 as Benign).
#
#--- FEATURE IMPORTANCE ---
# Top 10 features used by the model:
#              Feature  Importance
# 3      FlowBytesSent    0.225614
# 5  FlowBytesReceived    0.209809
# 2           Duration    0.176204
# 0         SourcePort    0.136081
# 6   FlowReceivedRate    0.127383
# 4       FlowSentRate    0.101609
# 1    DestinationPort    0.023300
#
#
#
#
# We can see that even without calculated features the model performs well (98.12% accuracy).
# That means we can add reward or punishment for the FN reaction on Malicious DoH because it is the most important for us to detect it correctly.