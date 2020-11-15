# Supervised-Machine-Learning_5-Methods-to-Predict-Customer-Churns
One cup of fresh orange juice has 124 mg of vitamin C, which is 200% of the recommended daily intake of vitamin C for an adult. With this as (completely unrelated) motivation, your task is to build a model to predict whether a grocery store customer will Purchase Citrus Hill (CH) or Minute Maid (MM) orange juice.

## **Solution Roadmap**
### Model Performance Metric
After reviewing the question and exploring the dataset, this case can be categorized as supervised machine learning, particularly, it is a classification problem with binary classes (Purchase is labeled as “yes” and “no”). Since the goal is to find the best balance between False Positives and False Negatives to each class, AUC score would be most appropriate metric to the performance of the predictive model.
### Data Modeling Summary
- 7 features (DiscCH, DiscMM,SalePriceCH,SalePriceMM, PriceDiff, STORE & Store7) were removed from dataset due to highly correlation with PctDiscCH, PctDiscMM and StoreID;
- Randomly split the data to an 80-to-20 ratio: while 80% of the dataset will be used for k-fold cross validation, rest 20% will be used to measure the performance of tuned models;
- 15-fold cross validation was used to tune each model to mitigate overfitting risk;
- 5 different models were built: Decision Tree, Naïve Bayes, KNN, XGBoost, and Logistic Regression; 

## ModelResults

Algorithm | AUC_Train |	AUC_Test | F1 Score
--------- | --------- | -------- | --------
Decision Tree |	0.92 | 0.85  |	0.83
Naïve Bayes |	0.85	| 0.80 |	0.77
KNN	|0.89 |	0.84	|0.82
XGBoost	| 0.92	| 0.88	| 0.85
Logistic Regression |	0.91 |	0.88 |	0.85

## Business Indication
- Best Model: XGBoost & Logistic Regression
- Worst Model: Naïve Bayes
- Area Under Curve (AUC) is the area under the ROC curve which contains both True Positive Rate (sensitivity, recall) and False Positive Rate(specificity), the higher the AUC Score means the better the overall model performance is. Usually, AUC score around 0.8 and near to 0.9 indicates that the prediction is excellent. In this case, both XGBoost and Logistic Regression have the highest AUC test score of 0.88, therefore, those two algorithms all best represent the prediction result. Meanwhile, the worst performance among these 5 fine-tuned models is Naïve Bayes with AUC score of 0.80.
- Furthermore, F1 Score is also calculated to measure the Precision and Recall rate of the result. Similar to AUC score, the higher the F1 Score indicates the higher the precision and higher the recall rate are. Usually, F1 score around 0.8 indicates that the prediction is good. In this case, both XGBoost and Logistic Regression have the highest AUC test score of 0.85, therefore, those two algorithms all best represent the prediction result. Meanwhile, the worst performance among these 5 fine-tuned models is Naïve Bayes with F1 score of 0.77.
