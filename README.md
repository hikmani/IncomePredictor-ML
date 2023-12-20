# Predictive Analysis of Predicting Income Level Kaggle Competitive Competition
Ahmed Al Hikmani

## Problem Statement and Motivation
The objective of the project was to create a model that could predict whether a person's yearly income surpasses $50,000, using data from the 1994 census. The difficulty in this task involves precisely determining income categories based on various demographic and other factors.
This problem is important because it provides key insights into economic trends and social frameworks. Grasping how income is distributed and what affects it is vital for economic study, policy formulation, and specialized social initiatives. The application of machine learning in this particular context arises from the need for an analytical technique capable of processing complex, multifaceted data and detecting prominent patterns.


## Technologies and Libraries Used
- Python
- Pandas, NumPy, Scikit-learn, XGBoost

## Experimental Results:
- **XGBoost Classifier:** Achieved the highest AUC of 0.92696, and scored 0.92896 on Kaggle, indicating strong predictive capabilities.
- **Naive Bayes:** Recorded an AUC of 0.83075, and scored 0.82791 on Kaggle, indicating decent performance.
- **Decision Tree:** Demonstrated an AUC of 0.88039, and scored 0.89132 on Kaggle, performing better than Naive Bayes but not as well as XGBoost.

The Decision Tree classifier appears to outperform the Naïve Bayes model, but not the XGBoost model.The XGBoost model's exceptional performance highlights its capability to efficiently manage complex datasets with varied features.

## References:
Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996. (PDF)
