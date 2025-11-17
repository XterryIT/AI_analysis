import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# We are training only 2 modules and in the end we have a tree. First model telling it is DoH or Non-DoH, second model telling it is Benign or Malicious.

def Decisiion_Tree_Model(X_train, X_test, y_train, y_test):
    
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def Random_Forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)


    print("Classification Report:")
    print(classification_report(y_test, y_pred))





def main():
    
    #DoH vs Non-DoH) ---
    print("="*60)
    print("DoH vs Non-DoH Model Training")
    print("="*60)

    df_A = pd.read_csv('merged_sample_for_1_agent.csv')

    df_A.fillna(0, inplace=True) 


    X_A = df_A.drop(['Label'], axis=1) 
    y_A = df_A['Label']
    

    print("0=NonDoH, 1=DoH:")
    print(y_A.value_counts())

    # Готовим данные для Модели А
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.3, random_state=300)

    # Запускаем модели
    print('Decision Tree Model Results:')
    Decisiion_Tree_Model(X_train_A, X_test_A, y_train_A, y_test_A)
    print("-"*100)
    print('\nRandom Forest Model Results:')
    Random_Forest_model(X_train_A, X_test_A, y_train_A, y_test_A)
    print("-"*100)


    # Benign vs Malicious
    print("\n\n" + "="*60)
    print("Benign vs Malicious Model Training")
    print("="*60)
    
    df_B = pd.read_csv('merged_sample_for_2_agent.csv')

    df_B.fillna(0, inplace=True)


    X_B = df_B.drop(['Label'], axis=1) 
    y_B = df_B['Label']
    

    print("0=Benign, 1=Malicious:")
    print(y_B.value_counts())
    

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.3, random_state=300)

    print('Decision Tree Model Results:')
    Decisiion_Tree_Model(X_train_B, X_test_B, y_train_B, y_test_B)
    print("-"*100)
    print('\nRandom Forest Model Results:')
    Random_Forest_model(X_train_B, X_test_B, y_train_B, y_test_B)
    print("-"*100)


if __name__ == "__main__":
    main()
