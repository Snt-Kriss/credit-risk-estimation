from mlflow.client import MlflowClient
import mlflow
from mlflow.pyfunc import PyFuncModel
from pprint import pprint

class MlFlowHandler:
    def __init__(self)-> None:
        tracking_uri= "sqlite:///mlflow.db"
        self.client= MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self)-> None:
        try:
            experiments= self.client.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return 'Service returning experiments'
            
        except Exception as e:
            return 'Error calling mlflow'
        
    def get_production_model(self)-> PyFuncModel:
        model_name= 'svc-credit-risk-classifier'
        model= mlflow.pyfunc.load_model(model_uri= f"models:/{model_name}/production")

        return model
    