# Reservation Cancellation Forecasting

The objective: 
* Build a Machine Learning model to predict whether a reservation is likely to be cancelled,
* Develop an MLOps architecture designed for a production environment.

To accomplish this task, **Airflow** and **Mlflow** are used to build Machine Learning Pipelines, fully customizable and ready for a production environment.

## Code organization

```sh
.
├── README.md
├── airflow
│   ├── dags
│   │   ├── inference_pipeline.py
│   │   └── training_pipeline.py
├── artifacts
├── data
│   ├── features_store
│   ├── preprocessed
│   ├── hotel_bookings.parquet
│   └── sample_for_inference.parquet
├── mlflow
├── notebooks
│   ├── 0_exploratory_data_analysis.ipynb
│   └── 1_preprocessed_data_check.ipynb
├── requirements.txt
└── steps
    ├── condition_step.py
    ├── config.py
    ├── feature_engineering_step.py
    ├── inference_step.py
    ├── preprocess_step.py
    ├── train_step.py
    └── utils
        ├── _artifact.py
        └── data_classes.py
```

The repository is structured as follows:

* **Data Exploratory Analysis (EDA)** is performed on **notebooks**,
* Each stage of the Machine Learning process (**Preprocessing**, **Training**, **Inference**, etc...) is defined as a module designed to be implemented into a pipeline. They are all located in the *steps/* folder.
* **Airflow** and **Mlflow** are deployed locally within this repository.
* In the *data* folder is located the original dataset that was provided for this assignement, in addition of a sample for batch prediction. *data/features_store* and *data/preprocessed* are directories to store the data once processed by some stages of the pipelines, such as **preprocessing** or **features_engineering** steps.
* The same idea for *artifacts* that contains **encoders** generated during the **features_engineering** step.