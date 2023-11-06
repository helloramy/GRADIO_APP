## Introduction

In the fiercely competitive business landscape, companies strive to maximize their profits and revenue margins. One key area where industry players focus their resources is customer retention. Keeping existing customers happy and loyal is crucial for sustaining business growth and long-term success. The primary objective of this project is to build a robust machine learning model that predicts customer churn with high accuracy. By identifying customers who are likely to churn, companies can proactively implement targeted retention strategies, reducing churn rates, and boosting customer loyalty.

## Objectives

The objective at the end of this project is to deploy a machine learning model capable of predicting customer churn in real-time. The successful implementation of this project will equip the company with a powerful tool to predict customer churn accurately, empowering the company to take proactive measures thereby increasing customer retention.

### Specific Objectives

The following questions are addressed in detail during the analysis:

1. What is the overall churn rate in the dataset?

2. How does the churn rate vary between male and female customers (gender)?

3. Does the presence of additional services like Online Security, Tech Support, Device Protection, Streaming TV, or Streaming Movies influence customer churn rates?

4. Is there a correlation between customer tenure (length of subscription) and the likelihood of churn?

5. What is the impact of internet service type (DSL or Fiber optic) on customer churn?

## Structure

### Evaluation of Data Accuracy

Completeness: Missing values, Incomplete records

Accuracy: Outliers, Data entry errors, Duplicates.

Consistency: Inconsistent formats, Inconsistent categories or labels, Inconsistent representations

Relevancy: Irrelevant variables, Outdated data

### Surface Properties of Dataset Report

Name of Dataset: Telco Churn Dataset.

Abstract: The dataset contains comprehensive information about the characteristics and behaviors of customers, including details about whether or not they churn.

Data Type: Multivariate

Format Type: Matrix


## Content

We perform an independent samples test to determine if the apparent difference between customers who churn and customers who do not is significant for each of the numerical variables.

If the p-value is less than a chosen significance level (0.05), you can reject the null hypothesis and conclude that there is a statistically significant difference in the central tendencies of the predictor variable for customers who churn and those who do not.

In other words, the predictor variable has a statistically significant effect on the target variable.

Because of the deviation from normality and class imbalance, non-parametric test will be used.

Unlike the t-test, which compares means, the Mann-Whitney U test compares the distributions of two groups to determine if they have the same population distribution or if one tends to have larger values than the other.

In other words, the Mann-Whitney U test assesses whether the two groups differ significantly in terms of their central tendencies or location. It does not directly compare means, but it evaluates whether the two groups have the same underlying population distribution.



## REFERENCES

https://www.datacamp.com/blog/data-science-in-marketing-customer-churn-rate-prediction

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

https://www.datapine.com/blog/data-analysis-questions/

Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.