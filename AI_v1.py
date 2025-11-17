import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, confusion_matrix



def Decisiion_Tree_Model(X_train, X_test, y_train, y_test):
    
    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def Logistic_Regration_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=15000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def Random_Forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def KN_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=10)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

def SVC_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)







def main():
    df = pd.read_csv('merged_sample.csv')

    df.fillna(0, inplace=True)

    X = df.drop(['Label'], axis=1) # X - wszystkie kolumny opróć tych co przywidujemy 
    y = df['Label'] # y - kolumna którą przewidujemy

    # PRzygotowanie dannych testowych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=300)

    print("Decision Tree Model:")
    Decisiion_Tree_Model(X_train, X_test, y_train, y_test)
    print("-"*100)
    print("\nLogistic Regression Model:")
    Logistic_Regration_model(X_train, X_test, y_train, y_test)
    print("-"*100)
    print("\nRandom Forest Model:")
    Random_Forest_model(X_train, X_test, y_train, y_test)
    print("-"*100)
    print("\nK-Nearest Neighbors Model:")
    KN_model(X_train, X_test, y_train, y_test)
    print("-"*100)
    print("\nSupport Vector Classifier Model:")
    SVC_model(X_train, X_test, y_train, y_test)
    print("-"*100)



if __name__ == "__main__":
    main()
