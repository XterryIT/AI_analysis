import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, confusion_matrix


# General test for all models


def decision_tree_model(x_train, x_test, y_train, y_test):
    
    model = DecisionTreeClassifier()

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def logistic_regration_model(x_train, x_test, y_train, y_test):
    model = LogisticRegression(max_iter=15000)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def random_forest_model(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def kn_model(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=10)

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def svc_model(x_train, x_test, y_train, y_test):
    model = SVC(kernel='rbf', random_state=42)

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)







def main():
    df = pd.read_csv('merged_sample.csv')

    df.fillna(0, inplace=True)

    x = df.drop(['Label'], axis=1) # x - all columns except the ones we expect
    y = df['Label'] # y - column we expect

    # preparing test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=300)

    print("Decision Tree Model:")
    decision_tree_model(x_train, x_test, y_train, y_test)
    print("-"*100)
    print("\nLogistic Regression Model:")
    logistic_regration_model(x_train, x_test, y_train, y_test)
    print("-"*100)
    print("\nRandom Forest Model:")
    random_forest_model(x_train, x_test, y_train, y_test)
    print("-"*100)
    print("\nK-Nearest Neighbors Model:")
    kn_model(x_train, x_test, y_train, y_test)
    print("-"*100)
    print("\nSupport Vector Classifier Model:")
    svc_model(x_train, x_test, y_train, y_test)
    print("-"*100)



if __name__ == "__main__":
    main()
