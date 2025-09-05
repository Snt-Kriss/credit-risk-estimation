import pandas as pd

def test_data_loading():
    df= pd.read_csv("./data/german_credit_data.csv")
    assert not df.empty, "Dataset should not be empty"
    assert "Risk" in df.columns, "Dataset must contain risk column"