# Credit Risk Estimation Microservice

Lending is one of the most important activities of the finance industry. Lenders provide loans to borrowers in exchange for the promise of repayment with interest. The lender makes a profit only if the borrower pays off the loan, thus asking two primary questions:

    - How risky is the borrower?
    - Given the risk, should we lend them?

Algorithms can be trained on data samples of consumer data to predict default of a loan borrower. The underlying trends can be assessed with algorithms and continuously analyzed to detect trends that might influence lending and underwriting risk in the future.

The goal of this project is to build a machine learning microservice using MLOps tools to predict the probability that a loan will default.
The data preparation, cleaning process and model selection processes can be found in the notebooks folder. 

## Architecture
Since we are working with dynamic requests, the microservice architecture is used, allowing to build a service that has the sole focus of retrieving the model from the model store and perform the requested inference. 

The microservice is built based on the following requirements:
    - The prediction should be rendered and accessible via a web-based dashboard
    - Users will be able to key in their information requested, and the system will classify them whether they are risky or not.

***Model Management***: MLflow tool has been used as the model artifact and metadata storage layer in the ML system. The ml system checks for models available for use in production in the MLflow server, retrieves it, and caches the model for use and reuse during the session if desired.

***Model Serving***: FastAPI web framework has been used to wrap the serving logic. The serving logic is in the serve/src folder.

***Continuous Intergration***: Every change to data, code or model is automatically tested so that the pipeline does not break. The model recall metric is set to at least 85% to ensure reduction of Type II error. The tests are in the tests folder. Github actions is used to set up the CI pipeline.

### Quick glance at the model results

| Model                  | Recall score   |
|------------------------|----------------|
| Support Vector Machine | 0.964          |
| Random Forest          | 0.950          |

**Production model used**: Support Vector Machine
**Metrics used**: Recall

### Metrics used explained
The primary objective of credit risk estimation is to minimize the risk of credit default for the institution/lender. Recall, or type II error is focused on the **False Negatives**, that is, predicting "not risky" when the applicant is actually "risky". A low recall means many false negatives, thus risky customers are incorrectly classified as good. 
Lending a risky borrower is very costly, leading to direct financial loss to the lending institution. 

In contranst, precision, or Type I error, leads to rejecting a good customer, but no actual financial loss. Recall has been prioritized to ensure that risky borrowers are caught by using ensemble models to train different algorithms, and tuning the model using GridSearchCV.

## Running the app locally
1. Initialize git repository

    ```bash
    git init
    ```

2. Clone the project

    ```bash
    git clone https://github.com/Snt-Kriss/credit-risk-estimation.git
    ```

3. Enter the project directory

    ```bash
    cd credit-risk-estimation
    ```

4. Create and activate a virtual environment (choose one)

   ```bash
   python -m venv .venv
   source .venv/bin/activate

5. Install dependencies using uv
    ```bash
    uv pip install -r pyproject.toml
    ```

6. Running the training pipeline:
    ```bash
    python training/classifier.py
    ```

8. Run the mlflow ui
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
9. Run the FastAPI app
    ```bash
    uvicorn serve.src.app:app --reload
    ```

10. Health check- Run this url to postman/web browser
    http://localhost:8000/health

11. Prediction Example:
    Json example:
                {
                    "Age": 67,
                    "Sex": "male",
                    "Job": 2,
                    "Housing": "own",
                    "SavingAccounts": "NA",
                    "CheckingAccount": "little",
                    "CreditAmount": 1169,
                    "Duration": 6,
                    "Purpose": "radio/TV"
                }

    URL: http://localhost:8000/predict

    Output: 
                {
                    "prediction": 1
                }