import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def CreateModel(data):
    target = "diagnosis"
    x = data.drop([target], axis=1)
    y = data[target]

    # data scalling 
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # splitting the data 
    x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2, random_state=2025)

    # train
    model = XGBClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, scaler, x_test, y_test   

def TestModel(model, x_test, y_test, show_scatter:bool, show_cm:bool = False):
    predictions = model.predict(x_test)
    acc_score = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='binary')
    summary = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"accuracy score : {acc_score*100:.2f}%")
    print(f"recall score : {recall*100:.2f}%")
    print("summary : \n", summary)
    
    if show_cm:
        sns.heatmap(cm, annot=True , fmt='d', cmap='cool')
        plt.title("confusion matrix")
        plt.show()

    if show_scatter:  
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))  
        plt.scatter(range(30), predictions[:30], s=550, alpha=0.5 ,  c='violet', label='predictions')
        plt.plot(range(30), y_test[:30], linewidth=4, c='darkviolet', label='true')
        plt.xlabel("Sample")
        plt.ylabel("Diagnosis")
        plt.title("Target Vs Predictions")
        plt.legend()
        plt.show()

def GetCleanData():
    data = pd.read_csv("data.csv", on_bad_lines='skip')
    # dropping useless columns
    data.drop(columns=["Unnamed: 32", "id"], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({"M": 1, "B": 0})
    return data

def main():
    data = GetCleanData()
    model, scaler, x_test, y_test = CreateModel(data)
    TestModel(model, x_test, y_test, show_scatter=True, show_cm=True)
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":

    main()




