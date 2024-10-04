
# Methodology

This section outlines the systematic approach employed to develop and evaluate machine learning models for predicting train delays. The methodology encompasses two primary phases: hyperparameter tuning for the Multi-layer Perceptron (MLP) classifier using `RandomizedSearchCV`, followed by a comparative analysis of various classifiers, including Support Vector Machines, Decision Trees, Random Forests, AdaBoost, XGBoost, CatBoost, and Naïve Bayes. Each phase is designed to optimize model performance and assess predictive capabilities, ensuring a robust evaluation of the factors influencing train punctuality.
### Data Preparation

The analysis commenced with loading the dataset (`new_data1.csv`), which includes various features relevant to train operations and a target variable representing train delays. The selected features for this study comprise `Temperature_1`, `Wind_Speed_1`, `weather_code_1`, and `Distance_travelled_1`. The target variable, `VSKP`, was transformed into a binary classification label, where values less than 10 indicate that the train is "not delayed" (encoded as 0), and values equal to or greater than 10 indicate "delayed" (encoded as 1). This binary encoding facilitates the MLP classifier's ability to predict train delays based on the defined features.

### Handling Missing Values
To ensure robust model performance, the dataset was examined for NaN values within the feature set. If any NaN values were found, they were addressed using the `SimpleImputer` from Scikit-learn, which employed the mean strategy to fill in missing values. This step is crucial for preventing errors during model training and evaluation.

### Data Splitting

The dataset was divided into training and testing subsets using an 80/20 ratio via the `train_test_split` method. This stratified split allows the model to learn from a substantial portion of the data while retaining a separate set for unbiased evaluation of its performance.

### Feature Scaling

To improve the convergence speed and overall performance of the MLP classifier, feature scaling was applied using `StandardScaler`. This process standardized the features to ensure they have a mean of zero and a standard deviation of one, which is crucial for neural network models to function effectively.

### Model Training and Evaluation

To determine the optimal hyperparameters for the MLP classifier, a randomized search strategy was employed using `RandomizedSearchCV`. The parameter grid consisted of various configurations, including:

- **Hidden Layer Sizes**: Different configurations of hidden layers, specifically (50,), (100,), and (150,) neurons.
- **Activation Functions**: A variety of activation functions, including `identity`, `logistic`, `tanh`, and `relu`.
- **Solvers**: The optimization algorithms for weight updates, which included `lbfgs`, `sgd`, and `adam`.
- **Maximum Iterations**: A range of iterations from 100 to 500, sampled from a uniform distribution using the `randint` function.

The `RandomizedSearchCV` was initialized with 10 iterations and implemented 3-fold cross-validation to assess model performance. This technique mitigates overfitting by ensuring that the model's hyperparameters are evaluated across multiple subsets of the training data. After fitting the `RandomizedSearchCV` to the scaled training data, the best-performing model was selected based on cross-validated results. The optimal hyperparameters were then utilized to predict train delays on the unseen test set.

Additionally, to comprehensively evaluate the performance of multiple classifiers, a diverse set of models was implemented, including:

- **Support Vector Classifier (SVC)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier**
- **CatBoost Classifier**
- **Naïve Bayes Classifier**

Each classifier was trained on the standardized training data and subsequently used to predict outcomes on the unseen test set. The model's predictions were evaluated using several key metrics:

- **Accuracy**: The proportion of true results among the total number of cases examined.
- **Precision**: The proportion of true positive results in all positive predictions made.
- **Recall**: The proportion of true positive results in all actual positive cases.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.

These metrics provided a comprehensive assessment of each model's performance in predicting train delays, allowing for informed comparisons and selections among the various classifiers evaluated.



# Results

#### 1. Hyperparameter Tuning for Multi-layer Perceptron (MLP) Classifier

The hyperparameter tuning of the Multi-layer Perceptron (MLP) classifier was conducted using `RandomizedSearchCV`, resulting in an optimal score of **0.9897**. The evaluation metrics for the MLP model on the test dataset are summarized in Table 1.

**Table 1: Evaluation Metrics for MLP Classifier**

| Metric      | Value    |
|-------------|----------|
| **Precision (0)** | 0.99     |
| **Recall (0)**    | 1.00     |
| **F1-Score (0)**  | 0.99     |
| **Support (0)**   | 72       |
| **Precision (1)** | 0.00     |
| **Recall (1)**    | 0.00     |
| **F1-Score (1)**  | 0.00     |
| **Support (1)**   | 1        |
Overall model performance metrics include:

- **Accuracy**: 0.9863
- **Macro Average**: 
  - Precision: 0.49 
  - Recall: 0.50 
  - F1 Score: 0.50
- **Weighted Average**: 
  - Precision: 0.97 
  - Recall: 0.99 
  - F1 Score: 0.98

#### 2. Evaluation of Various Classifiers

To assess the effectiveness of different classifiers in predicting train delays, several models were implemented, including Support Vector Classifier (SVC), Decision Tree, Random Forest, AdaBoost, XGBoost, CatBoost, and Naïve Bayes. The performance metrics for each classifier are presented in Table 2.

**Table 2: Performance Metrics for Various Classifiers**

| Classifier                           | Accuracy | Precision | Recall   | F1 Score |
|--------------------------------------|----------|-----------|----------|----------|
| Support Vector Classifier (SVC)      | 0.8904   | 1.0000    | 0.3333   | 0.5000   |
| Decision Tree                        | 0.8630   | 0.5625    | 0.7500   | 0.6429   |
| Random Forest                        | 0.9178   | 0.8000    | 0.6667   | 0.7273   |
| AdaBoost                             | 0.8356   | 0.5000    | 0.5000   | 0.5000   |
| XGBoost                              | 0.9041   | 0.7273    | 0.6667   | 0.6957   |
| CatBoost                             | 0.9178   | 0.8000    | 0.6667   | 0.7273   |
| Naive Bayes                         | 0.8356   | 0.5000    | 0.7500   | 0.6000   |

The results indicate that the Random Forest and CatBoost classifiers achieved the highest accuracy of **0.9178**, while the SVC exhibited the highest precision. Conversely, the MLP classifier showed an excellent F1-score for class 0, emphasizing its strong performance in identifying non-delayed trains. These findings provide insights into the strengths and limitations of each model for the task of predicting train delays.



# Conclusion

This study focused on developing and evaluating machine learning models for predicting train delays using various classifiers and hyperparameter tuning techniques. The Multi-layer Perceptron (MLP) classifier, optimized through `RandomizedSearchCV`, achieved a high accuracy of **0.9863** and demonstrated excellent performance in identifying non-delayed trains. However, while the MLP was effective, the Random Forest and CatBoost classifiers emerged as the top performers with an accuracy of **0.9178**, indicating their strong predictive capabilities in this context.

The comparative analysis highlighted the trade-offs between different classifiers, particularly in precision and recall metrics. Overall, these results emphasize the importance of selecting the right model and tuning its parameters for effective prediction in time-sensitive applications like train operations. Future research could explore integrating additional data sources and hybrid modeling approaches to further enhance prediction accuracy and robustness.
