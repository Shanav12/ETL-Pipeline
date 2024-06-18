# ETL-Pipeline

## Inspiration:
I wanted to embark on a journey to understand what factors can play in to whether a company will retain a customer or not. To do so, I found this dataset: https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis and uploaded it to MySQL. To construct this pipeline, I extracted the customer table through PySpark and began performing an analysis to learn about consumer behavior.

## Tech Stack:
- Spark
- Python
- MySQL
- Google BigQuery

## Overview: 
I transformed the data in PySpark through filtering out rows containing null values, grouped all unique customers, engineered a purchases feature, and loaded it to Google BigQuery. After loading the data, I engineered a logistic regression model to predict customer retainment and was able to achieve an accuracy rate of 75%. Through this model, I was able to learn that customer age, size of purchases, order quantity, and total number of orders are important factors in determining whether a company will retain a customer.
