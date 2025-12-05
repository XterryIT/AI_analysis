import pandas as pd
import joblib
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'rf_model_multiclass.joblib'
FEATURES_PATH = 'model_features.joblib'

# Path to your new data file for testing (MUST have a 'Label' column)
NEW_DATA_PATH = 'ultimate_sample.csv' 

def print_report(y_true, y_pred):
    """
    Prints a formatted report, similar to the one used during training.
    """
    print("\n" + "="*60)
    print("TESTING RESULTS ON NEW DATA")
    print("="*60)
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    
    # 2. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    labels = [0, 1, 2]
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Print headers
        print(f"{'':>15} | {'Pred: Non-DoH':>15} | {'Pred: Benign':>15} | {'Pred: Malicious':>15}")
        print("-" * 70)
        
        row_names = ['Act: Non-DoH', 'Act: Benign', 'Act: Malicious']
        for i, row in enumerate(cm):
            print(f"{row_names[i]:>15} | {row[0]:>15} | {row[1]:>15} | {row[2]:>15}")
            
    except Exception as e:
        print(f"Error building confusion matrix (check class labels): {e}")

    # 3. Detailed Report
    print("\n--- Detailed Report ---")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], 
                                target_names=['Non-DoH', 'Benign', 'Malicious'], digits=4))





def main():
    print("### STARTING MODEL EVALUATION ###")


    # 1. Load Model and Features
    print(f"[1] Loading model and feature list...")
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # 2. Load Data
    print(f"[2] Reading file '{NEW_DATA_PATH}'...")
    try:
        if NEW_DATA_PATH.endswith('.csv'):
            df = pd.read_csv(NEW_DATA_PATH)
        else:
            df = pd.read_excel(NEW_DATA_PATH)
        
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return

    # Check for the ground truth column
    if 'Label' not in df.columns:
        print("CRITICAL ERROR: Column 'Label' is missing in the file.")
        print("Ground truth labels are required for quality testing.")
        return

    # 3. Data Preparation (Aligning columns)
    print("[3] Preparing data...")
    X_new = pd.DataFrame()
    for col in feature_names:
        if col in df.columns:
            X_new[col] = df[col]
        else:
            # If a feature from training is missing in new data, fill with 0
            X_new[col] = 0 
    
    # Strictly ensure column order matches training
    X_new = X_new[feature_names]
    y_true = df['Label'] # Ground truth labels

    # 4. Prediction
    print("[4] Model is generating predictions...")
    y_pred = model.predict(X_new)

    try:
        df = pd.read_csv('ultimate_sample.csv')
    except FileNotFoundError:
        print("ERROR: File 'ultimate_sample.csv' not found!")
        return 

    df.fillna(0, inplace=True)

    X = df.drop(['Label'], axis=1) 
    y = df['Label']

    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")



    # 5. Output Results
    print_report(y_true, y_pred)

if __name__ == "__main__":
    main()


# I tested pipiline on two different datasets:
#
# (We have only 19000 Bening samples)
# (I worte custom SMOTE to extend samples in benign file to 100 000)
#
# 1) merged_sample.csv - formed with 10 000 samples from each file 
# ultimate formed with 30 000 samples from each file
#
# In conclusion what we have:
#
#
# !!!!!!!!!!!!!!!!!TRAINING MODEL ON merged_sample.csv:!!!!!!!!!!!!!!!!!!!!!!!1
# Label
# 0    20000
# 1    40000
# 2    20000
# Name: count, dtype: int64
# (0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)
#
#
# Random Forest Model Results:
#
# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 97.62%
#
#
#
# ============================================================
# CONFUSION MATRIX
# ============================================================
#                 |   Pred: Non-DoH |    Pred: Benign | Pred: Malicious
# ---------------------------------------------------------------------
#    Act: Non-DoH |            5849 |             120 |               8
#     Act: Benign |             154 |           11688 |             132
#  Act: Malicious |              25 |             133 |            5891
# ---------------------------------------------------------------------

# MALICIOUS TRAFFIC ANALYSIS (Security Focus):
# - True Positives: 5891
# - False Negatives (Missed Attacks): 158
#   --> Breakdown: 25 predicted as Non-DoH + 133 as Benign.
#
#
#
#
# !!!!!!!!!!!!!!!!WORKING MODEL ON ultimate_sample.csv:!!!!!!!!!!!!!!!!!
# Label
# 0    30000
# 1    60000
# 2    30000
# Name: count, dtype: int64

# ============================================================
# TESTING RESULTS ON NEW DATA
# ============================================================
# Overall Accuracy: 65.85%


# --- Confusion Matrix ---
#                 |   Pred: Non-DoH |    Pred: Benign | Pred: Malicious
# ----------------------------------------------------------------------
#    Act: Non-DoH |           20510 |            9157 |             333
#     Act: Benign |            1037 |           30965 |           27998
#  Act: Malicious |             785 |            1672 |           27543
#
#
#
#
#
# 2) I took only legitimate samples (without SMOTE generation) 
# merged_sample.csv - formed with 5 000 samples from each file 
# ultimate formed with 10 000 samples from each file
#
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!TRAINING MODEL ON merged_sample.csv:!!!!!!!!!!!!!!!!!!!!!!!
#  
#    Label
# 0     5000
# 1    10000
# 2     5000
# Name: count, dtype: int64
# (0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)


# Random Forest Model Results:

# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.05%



# ============================================================
# CONFUSION MATRIX
# ============================================================
#                 |   Pred: Non-DoH |    Pred: Benign | Pred: Malicious
# ---------------------------------------------------------------------
#    Act: Non-DoH |            1421 |              25 |               2
#     Act: Benign |              22 |            2996 |              39
#  Act: Malicious |               0 |              29 |            1466
# ---------------------------------------------------------------------

# MALICIOUS TRAFFIC ANALYSIS (Security Focus):
# - True Positives: 1466
# - False Negatives (Missed Attacks): 29
# ============================================================
#
#
#
#
#
# !!!!!!!!!!!!!!!!!WORKING MODEL ON ultimate_sample.csv:!!!!!!!!!!!!!!!!!
# 
# Label     
# 0    10000
# 1    20000
# 2    10000
# Name: count, dtype: int64
# (0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)


# ============================================================
# TESTING RESULTS ON NEW DATA
# ============================================================
# Overall Accuracy: 84.54%

# --- Confusion Matrix ---
#                 |   Pred: Non-DoH |    Pred: Benign | Pred: Malicious
# ----------------------------------------------------------------------
#    Act: Non-DoH |            4645 |            5188 |             167
#     Act: Benign |             147 |           19662 |             191
#  Act: Malicious |              45 |             445 |            9510
#
#
# In conclusion I want to said that model works better with more legitimate data but still have a problems.














