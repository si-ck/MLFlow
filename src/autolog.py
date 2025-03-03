import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='si-ck', repo_name='my-first-repo', mlflow=True)



mlflow.set_tracking_uri("https://dagshub.com/si-ck/MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 5
n_estimators = 8

mlflow.autolog()
mlflow.set_experiment("exp1 - autolog")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    
    #creating confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d', cmap='Blues',xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion-matrix.png")

    # ONLY THE NOTEBOOK IS NOT LOGGED USING AUTOLOG
    
    mlflow.log_artifact(__file__)
    
    # tagging, SHOULD BE DONE EXPLICITLY AS AUTOLOG DOESNT SET TAGS
    mlflow.set_tags({"Author":"sik","proj":'wine_classic=fication'})
    

    print(accuracy)