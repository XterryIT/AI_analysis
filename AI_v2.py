import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# We are training only 2 modules and in the end we have a tree. First model telling it is DoH or Non-DoH, second model telling it is Benign or Malicious.

def decision_tree_model(x_train, x_test, y_train, y_test):
    
    model = DecisionTreeClassifier(random_state=42)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)


    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def random_forest_model(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)

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

    df_a = pd.read_csv('merged_sample.csv')

    df_a.fillna(0, inplace=True)


    x_a = df_a.drop(['Label'], axis=1)
    y_a = df_a['Label']
    

    print("0=NonDoH, 1=DoH:")
    print(y_a.value_counts())

    # Готовим данные для Модели А
    x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(x_a, y_a, test_size=0.3, random_state=300)

    # Запускаем модели
    print('Decision Tree Model Results:')
    decision_tree_model(x_train_a, x_test_a, y_train_a, y_test_a)
    print("-"*100)
    print('\nRandom Forest Model Results:')
    random_forest_model(x_train_a, x_test_a, y_train_a, y_test_a)
    print("-"*100)


    # Benign vs Malicious
    print("\n\n" + "="*60)
    print("Benign vs Malicious Model Training")
    print("="*60)
    
    df_b = pd.read_csv('merged_sample.csv')

    df_b.fillna(0, inplace=True)


    x_b = df_b.drop(['Label'], axis=1)
    y_b = df_b['Label']
    

    print("0=Benign, 1=Malicious:")
    print(y_b.value_counts())
    

    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_b, y_b, test_size=0.3, random_state=300)

    print('Decision Tree Model Results:')
    decision_tree_model(x_train_b, x_test_b, y_train_b, y_test_b)
    print("-"*100)
    print('\nRandom Forest Model Results:')
    random_forest_model(x_train_b, x_test_b, y_train_b, y_test_b)
    print("-"*100)


if __name__ == "__main__":
    main()
