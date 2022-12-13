# Predict the survival rate of heart failure patients.

As part of this project, I have to predict patients’ s survival based on their historical medical reports using AutoML and Hyperdrive.

## Project Set Up and Installation

- AutoML experiement configuration in Jupyter notekbook -> automl.ipynb
- setup Hyperdrive settings and configuration -> hyperparameter_tuning.ipynb
- Created traiing folder and place the train.py file
- Clean data and passing parameters -> train.py
- Added conda package dependencies file conda_dependencies.yml

## Dataset
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Overview
The dataset userd here is taken from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

The dataset provided in csv file contain 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| age | Age of patient | Years (40-95) |
| anaemia | Decrease of red blood cells or hemoglobin | boolean (0=No, 1=Yes) |
| creatinine-phosphokinase | Level of the CPK enzyme in the blood | boolean mcg/L |
| diabetes | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| ejection_fraction | Percentage of blood leaving the heart at each contraction | Percentage |
| high_blood_pressure | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| platelets | Platelets in the blood | kiloplatelets/mL	|
| serum_creatinine*| Level of creatinine in the blood | mg/dL |
| serum_sodium | Level of sodium in the blood | mEq/L |
| sex | Female (F) or Male (M) | Binary (0=F, 1=M) |
| smoking | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| time | Follow-up period | Days |
| DEATH_EVENT | Whether the patient died during the follow-up period | Boolean (0=No, 1=Yes) |

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

