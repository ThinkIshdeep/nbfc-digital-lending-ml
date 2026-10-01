# NBFC ML Project — Easy Study Guide

## 1. One-line project explanation
"I used official RBI digital-lending data, cleaned it in Python, made visualisations, and used machine learning to study and predict observable NBFC lending patterns."

## 2. What is an NBFC?
An NBFC is a Non-Banking Financial Company. It provides financial services such as loans but is not a traditional bank.

## 3. What is loan disbursement?
Disbursement means the actual loan money released to the borrower.

## 4. What is loan count?
The number of loans recorded.

## 5. What is average ticket size?
Average amount per loan.

## 6. Why Python?
Python makes it easy to clean data, calculate values, create graphs and run machine-learning models.

## 7. Why Pandas?
Pandas is used to read, clean and analyse tables.

## 8. Why machine learning?
Machine learning helps find patterns and make predictions from existing data.

## 9. Classification
Our classification question is:
"Will the next observed period's digital disbursement be higher than the current period?"

Output: Yes / No.

## 10. Regression
Regression predicts a number.
Our target is next observed digital disbursement in ₹ crore.

## 11. Random Forest
Random Forest combines many decision trees. Each tree makes a prediction and the combined result is used.

## 12. K-Means
K-Means groups similar observations. We used it to create 3 data-driven NBFC segments.

## 13. Isolation Forest
It identifies observations that look different from most other observations. It is an investigation tool, not proof of fraud.

## 14. What is F1 score?
F1 combines precision and recall into one score. Higher is better for the evaluated classification task.

## 15. What is RMSE?
RMSE measures how far numerical predictions are from actual values. Lower is better.

## 16. What is R²?
R² shows how much variation in the target is explained by the model. It can be below 0 when a model performs worse than a simple baseline.

## 17. Why did we use GroupKFold?
Because several rows belong to the same NBFC. GroupKFold keeps an NBFC's observations together when creating train/test folds, reducing leakage risk.

## 18. What is the most important limitation?
The public dataset does not contain private internal NBFC analytics-system data or borrower-level information. So our project does not prove that analytics caused lending changes.

## 19. What if teacher asks: "Can this approve a loan?"
Answer:
"No. Our public dataset does not have borrower income, credit score, repayment history or approval/rejection data. Our model studies aggregate digital-lending patterns."

## 20. What if teacher asks: "Why not deep learning?"
Answer:
"The available dataset is small and aggregate, so simpler machine-learning models are more suitable and easier to evaluate honestly."

## 21. What did you actually do?
1. Collected/used RBI public data.
2. Cleaned the dataset.
3. Created useful variables.
4. Made Python visualisations.
5. Created a transition dataset.
6. Trained classification and regression models.
7. Used K-Means for segmentation.
8. Used Isolation Forest for unusual-pattern detection.
9. Evaluated the models.
10. Interpreted the results.

## 22. Results to remember
- Cleaned dataset: 305 NBFC-period rows.
- ML transitions: 135.
- NBFC groups in ML: 51.
- Best classifier: Random Forest Classifier.
- Best classifier F1: 0.809.
- Best regression model by RMSE: Gradient Boosting Regressor.
- Regression RMSE: ₹1112.69 crore.
- K-Means clusters: 3.
- Potentially unusual latest observations: 6.

## 23. 30-second presentation script
"My project is about data analytics and machine learning for digital loan disbursement in NBFCs. I used public RBI data and first cleaned it using Python and Pandas. Then I created graphs for disbursement, loan count and average ticket size. After that I created machine-learning models. Classification predicts whether the next observed disbursement increases, regression estimates the next amount, K-Means groups NBFCs by lending behaviour, and Isolation Forest flags unusual patterns. The main limitation is that the public data is aggregate, so the models do not represent private borrower-level credit decisions."
