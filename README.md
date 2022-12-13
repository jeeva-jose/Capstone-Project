# Predict the survival rate of heart failure patients.

As part of this project, I have to predict patients’ s survival based on their historical medical reports using AutoML and Hyperdrive.

## Project Set Up and Installation

- AutoML experiement configured in Jupyter notekbook -> automl.ipynb
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
My goal is classify patients based on above 12 features and predict the survival target column as DEATH_EVENT. 

### Access
The dataset downloaded from Kaggle website and place the csv file in my GitHub public repository.

[GitHub](https://raw.githubusercontent.com/jeeva-jose/Capstone-Project/main/heart_failure_clinical_records_dataset.csv)

Registerd Dataset
 ![Dataset](/Dataset.png "Register Dataset")

## Automated ML

```

automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}

```
```
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

- compute_target: A compute target is a designated compute resource or environment where you run your training script or host your service deployment.
- task : what task needs to be performed , classification
- training_data : the data on which we need to train the autoML.
- label_column_name : the column name in the training data which is the output label.
- iterations : the number of iterations we want to run AutoML.
- primary_metric : the evaluation metric for the models
- n_cross_validations : n-fold cross validations needed to perform in each model
- experiment_timeout_minutes : the time in minutes after which autoML will stop.
- featurization: auto defines whether featurization step should be done automatically 

     


### Results
Run details and result attached here

accuracy : 0.849425287356322

 ![BestMetric](/Best%20Metric.png "Best Result")
 ![BestMetric](/BestModel.png "Best Result")

Enable deep learning can produce a better result.

## Hyperparameter Tuning

For this experiment I am using a Scikit-learn Logistic Regression model, parameter sampler using the parameters C and max_iter and chose discrete values with choice for both parameters.

**Parameter sampler**

In random sampling, hyperparameter values are randomly selected from the defined search space. Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance jobs

I chose discrete values with choice for both parameters, C and max_iter.

C is the Regularization while max_iter is the maximum number of iterations.

```
ps = RandomParameterSampling({
    '--C': choice(1.0, 0.1, 0.05),
    '--max_iter': choice(50,100,150)})
    
 ```
    
**Early stopping policy** 

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

```
 # Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
```

evaluation_interval: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.


### Results

Generated the best model and accuracy

 ![HyperDriveResult](/HyperDriveBestModel.png "HyperDriveResult")


increase the max_iter can be improve the performance of the model.


 ![HyperDriveResult](/HyperDriveFromUI.png "HyperDriveResult")

## Model Deployment

Created the experiment in AutoML and predicated the best model, register the model for the deployment. Here we need to pass the entry script and environment details are required during the deployment time. The help of Webservice library to deploy the selected model in to Azure container instance. 

scoring script :   It loads and de-serializes the model from its .pkl file in the init-method and it receives parameters sent to the Azure App Service, passes them into the machine learning model for scoring and then returns the resulting values in the run-method.

inference config : what the Python execution environment will be like: What package dependencies does it have 


 ![Endpoint](/Endpoint.png "Endpoint")
 ![Deployment](/Deployment%20details.png "Deployment details")
 ![Deployment](/Endpoint%20testing%20result.png "Deployment details")

## Screen Recording
https://clipchamp.com/watch/Fq733LWFlJR

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

