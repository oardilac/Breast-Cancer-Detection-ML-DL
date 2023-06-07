# Breast Cancer Diagnosis Project

This repository includes a Python-based breast cancer diagnosis project using several machine learning models. The dataset used in this project is `breast_cancer_dataset.csv`, which contains a series of medical predictor characteristics, as well as a target variable of diagnosis (M = malignant, B = benign).

## Project Structure

The project is divided into the following sections:

1. Data Exploration and Preprocessing
2. Data Visualization and Analysis
3. Data Preparation for Modeling
4. Modeling and Evaluation

## Requirements

To run this project, you'll need the following Python packages:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* tensorflow

You can install these packages using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Usage

The project script can be run in a Jupyter notebook or any other Python environment.

## File Descriptions

* `breast_cancer_dataset.csv`: This is the dataset used for the project.
* `pca_95.pkl`: This is the PCA object saved after fitting on the dataset.
* `modelsML.pkl`: This contains the VotingClassifier model fitted on the data.
* `best_score_model.pkl`: This contains the best performing model after hyperparameter tuning.
* `my_model.h5`: This contains the trained neural network model.

## Results

The results from the different machine learning models are displayed in terms of accuracy and confusion matrices. The best performing model's parameters are also printed out. The models include Support Vector Machines (SVM), Decision Trees, Random Forests, Gradient Boosting, K-Nearest Neighbors (KNN), and a Neural Network model.

The VotingClassifier model is also saved for future use, as well as the best performing model after hyperparameter tuning. A neural network model is also trained and saved.

## Contributing

Contributions to this project are welcome! Please submit a pull request or an issue if you'd like to add or suggest something.

License
This project is open source, under the [MIT License](LICENSE).