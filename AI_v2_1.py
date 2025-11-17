import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# We are training one model for 3 parametrs classification: DoH, Non-DoH Benign, Malicious.

def Decisiion_Tree_Model(X_train, X_test, y_train, y_test):
    
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))


def Random_Forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)


    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))





def main():
    
    print("#" * 20)
    print("Multiclass")
    print("#" * 20)

    df = pd.read_csv('merged_sample.csv')

    df.fillna(0, inplace=True)

    X = df.drop(['Label'], axis=1) 
    y = df['Label']

    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)

    print('Decision Tree Model Results:')
    Decisiion_Tree_Model(X_train, X_test, y_train, y_test)
    print("-"*100)
    print('\nRandom Forest Model Results:')
    Random_Forest_model(X_train, X_test, y_train, y_test)
    print("-"*100)

if __name__ == "__main__":
    main()
