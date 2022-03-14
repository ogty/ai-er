import os
import pickle
import subprocess

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier


def func_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    train_rows = ((df.attack_type == "norm") | (df.attack_type == "sqli"))
    df = df[train_rows]

    entropies = []
    closing_parenthesis = []
    for i in df["payload"]:
        entropies.append(H_entropy(i))

        if i.count(")"):
            closing_parenthesis.append(1)
        else:
            closing_parenthesis.append(0)

    df = df.assign(entropy=entropies)
    df = df.assign(closing_parenthesis=closing_parenthesis)

    rep = df.label.replace({"norm": 0, "anom": 1})
    df = df.assign(label=rep)

    return df


def H_entropy(query: str) -> float:
    prob = [float(query.count(c)) / len(query) for c in dict.fromkeys(list(query))]
    H = abs(sum([p * np.log2(p) for p in prob]))
    return H


# class for adjacing parameters
class ObjectiveDTC:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.trial._trial.Trial) -> np.float64:
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 64)
        }
        model = DecisionTreeClassifier(**params)
        scores = cross_validate(model, X=self.X, y=self.y, scoring="accuracy", n_jobs=-1)

        return scores["test_score"].mean()


def setup() -> None:
    pyload_train_path = "./data/payload_train.csv"
    pyload_test_path = "./data/payload_test.csv"

    # If there is no data folder, create one and download the data set.
    if not os.path.exists("data"):
        os.mkdir("data")

    subprocess.run(["curl", "-L", "-o", pyload_test_path, "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/payload_test.csv"])
    subprocess.run(["curl", "-L", "-o", pyload_train_path, "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/payload_train.csv"])

    # load train data
    df = pd.read_csv(pyload_train_path)
    df = func_preprocessing(df)

    # load test data and preprocessing
    test_data = pd.read_csv(pyload_test_path)
    test_data = func_preprocessing(test_data)

    # data splitting
    df_x = df[["length", "entropy", "closing_parenthesis"]]
    test_x = test_data[["length", "entropy", "closing_parenthesis"]]

    df_y = df[["label"]]
    test_y = test_data[["label"]]

    X_all = pd.concat([df_x, test_x])
    y_all = pd.concat([df_y, test_y])
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True, random_state=101)
        
    objective = ObjectiveDTC(X_train, y_train)
    study = optuna.create_study()
    study.optimize(objective, timeout=60)
    print(f"params: {study.best_params}")


    model = DecisionTreeClassifier(
        criterion=study.best_params["criterion"],
        max_depth=study.best_params["max_depth"]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Accuracy: {100 * accuracy_score(y_test, pred):.5f} %")
    print(confusion_matrix(y_test, pred))


    # save model
    if not os.path.exists("models"):
        os.mkdir("models")

    with open("./models/model.pkl", "wb") as f:
        pickle.dump(model, f)


def is_sql_injections(model, url: str) -> bool:
    query = url.split("?")[1]
    entropy = H_entropy(query)
    query_length = len(query)

    if "%29" in query:
        closed_pararenthesis = True
    else:
        closed_pararenthesis = False

    result = model.predict([[
        query_length,
        entropy,
        closed_pararenthesis
    ]])

    if result[0]:
        return True
    else:
        return False


if __name__ == "__main__":
    data = input("""
    Please enter one of the following

     - setup  : Download the dataset and save the model. Takes about a minute.
     - example: Show example
     - url    : Outputs whether the input URL endpoint is subject to sql injection.

    :""")
    if data == "setup":
        setup()
    elif data == "example":
        with open("./models/model.pkl", "rb") as f:
            model = pickle.load(f)

        # URL is appropriate.
        sample_url = "https://www.xyzzy.com/watch?v=xxx&t=xxx"
        sample_url2 = "https://www.xyzzy.com/post?v=INSERT+INTO+users+%28name%2Cmail_address%29+VALUES+%28%27%E5%B1%B1%E7%94%B0%E5%A4%AA%E9%83%8E%27%2C+%28select+user_password+from+user_password+where+id%3D%271%27+limit+0%2C1%29%29+--+%E3%81%8A%E3%82%8F%E3%82%8A%27%2C%27aaa%40co.jp%27%29"

        print(f"URL1: {sample_url}")
        print(f"URL2: {sample_url2}")

        print(is_sql_injections(model, sample_url))
        print(is_sql_injections(model, sample_url2))
    else:
        with open("./models/model.pkl", "rb") as f:
            model = pickle.load(f)
            print(is_sql_injections(model, data))
