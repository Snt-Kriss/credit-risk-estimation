from pydantic import BaseModel
import pandas as pd

class ClassifierRequest(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    SavingAccounts: str
    CheckingAccount: str
    CreditAmount: int
    Duration: int
    Purpose: str