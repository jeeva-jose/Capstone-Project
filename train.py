from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.dataset import Dataset

def clean_data(data):

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")

    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    #dataset = Dataset.get_by_name(ws, name='heart_failure_clinical_records')
    my_dataset = 'https://raw.githubusercontent.com/jeeva-jose/Capstone-Project/main/heart_failure_clinical_records_dataset.csv'
    ds = Dataset.Tabular.from_delimited_files(my_dataset) 
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    ### YOUR CODE HERE ###a

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    
    os.makedirs('outputs',exist_ok = True)
    
    joblib.dump(value=model, filename='outputs/model.pkl')
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()