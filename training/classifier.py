import pandas as pd
import numpy as np
import mlflow
import json
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from mlflow.client import MlflowClient
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def load_data(path: str)-> DataFrame:
    df= pd.read_csv(path)

    return df

def encode_categorical_values(data: DataFrame)-> DataFrame:
    features= data.loc[:, data.columns != 'Risk']
    categorical_cols= features.select_dtypes(include='O').columns.tolist()

    df= data.copy()

    df['SavingAccounts']= df['SavingAccounts'].fillna('unknown')
    df['CheckingAccount']= df['CheckingAccount'].fillna('unknown')

    label_encoders= {}

    for col in categorical_cols:
        le=LabelEncoder()
        df[col]= le.fit_transform(df[col])
        label_encoders[col]= le

    return df

def convert_infinite_values(df: DataFrame)-> DataFrame:
    df= df.replace([np.inf, -np.inf], np.nan)

    return df

def train_test_split_data(df: DataFrame, test_size: int)-> tuple[DataFrame, DataFrame, Series, Series]:
    df['Risk']= df['Risk'].replace({'good': 1, 'bad': 0})
    X= df.drop(columns=['Risk'])
    y= df['Risk']

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_size, random_state=42)
    scaler= StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(X_train: DataFrame, y_train: Series)-> SVC:
    model= SVC(C=1.3, kernel='rbf')
    model.fit(X_train, y_train)

    return model

def predict(model: SVC, X_test: DataFrame)-> Series:
    y_pred= model.predict(X_test)
    return y_pred

if __name__ == "__main__":
    import logging
    log_format= "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format= log_format, level= logging.INFO)

    tracking_uri= "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client= MlflowClient()
    logging.info("MlflowClient defined and set tracking uri")

    mlflow.set_experiment('credit-risk-estimator')
    logging.info("Experiment set")
    

    run_name= "svc-experiment"
    artifact_path= 'model'

    df= load_data("./data/german_credit_data.csv")
    logging.info("Data loaded successfully")

    with mlflow.start_run(run_name=run_name):
        logging.info("Started mlflow run")

        df_encoded= encode_categorical_values(df)
        logging.info("Encoded categorical values")

        final_data= convert_infinite_values(df_encoded)
        logging.info("Infinite values converted")

        X_train, X_test, y_train, y_test= train_test_split_data(final_data, test_size=0.2)
        logging.info("Split data into train/test data")

        model= train_model(X_train, y_train)
        logging.info("SVC model trained and fitted")

        y_pred= predict(model=model, X_test=X_test)

        acc= accuracy_score(y_test, y_pred)
        recall= recall_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)

        df_cm= pd.DataFrame(confusion_matrix(y_test, y_pred), columns=np.unique(y_test), index=np.unique(y_test))
        df_cm.index.name= 'Actual'
        df_cm.columns.name= 'Predicted'
        sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size': 16})
        plt.savefig('confusion_matrix.png') 

        metrics={
            "recall": recall
        }
        with open("metrics.json", 'w') as f:
            json.dump(metrics, f)

        model_name= 'svc-credit-risk-classifier'
        params= {
            'C':1.3,
            'kernel': 'rbf',
        }

        run_id= mlflow.active_run().info.run_id

        mlflow.sklearn.log_model(sk_model=model, name=artifact_path)
        logging.info("Model logged successfully")

        mlflow.log_params(params)

        mlflow.log_metrics({
            'accuracy': acc,
            'recall': recall,
            'precision': precision,
        })

        mlflow.log_artifact('confusion_matrix.png')
        logging.info("Metrics and artifacts logged")

        model_uri= f"runs:/{run_id}/{artifact_path}"

        model_details= mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info("Model registered")

        client.transition_model_version_stage(
            name= model_details.name,
            version= model_details.version,
            stage='production',
        )
        logging.info("Model transitioned to production stage")





