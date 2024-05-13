from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from login import username, password, project_id, table_id, dataset_id, bucket
from google.cloud import bigquery
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import numpy as np
import pandas_gbq as pd


class ETLPipeline:
    """
        We will take in a connection url in order to read in the MySQL table and load it into a PySpark Dataframe.
        Setting the configuration to the correct timezone to ensure the server runs as intended.
        Setting the connection properties using the jdbc driver and credentials.
    """
    def __init__(self, connection_url : str):
        self.spark = SparkSession.builder.config("spark.jars.packages", "mysql:mysql-connector-java:8.0.29").getOrCreate()
        self.spark.conf.set("spark.sql.session.timeZone", "UTC")
        self.connection_properties = {
            "user": username,
            "password": password, 
            "driver": "com.mysql.cj.jdbc.Driver", 
            "serverTimezone": "UTC"
        }
        self.connection_url = connection_url

    """
        Reading in the specified MySQL table into a PySpark dataframe that we'll be using for the model.
        Dropping all rows with null values to ensure the dataframe we use doesn't have missing data.
    """
    def table_to_dataframe(self, tablename : str) -> None:
        df = self.spark.read.jdbc(url=self.connection_url, table=tablename, properties=self.connection_properties)
        self.spark_df = df.na.drop()

    """
        Grouping all customer entries together to aggregate all data by each unique customer.
        Retaining only the Gender, Age, and Churn per customer and creating the number of purchases by each.
        Summing the quantity and price per customer and dropping all other columns.
        Renaming churn to label in order to utilize the Logistic Regression model.
    """
    def group(self) -> None:
        self.df = self.spark_df.groupBy("Customer_Name").agg(
            F.sum("Quantity").alias("Total_Quantity"), 
            F.sum("Product_Price").alias("Total_Price"), 
            F.first("Gender").alias("Sex"), 
            F.first("Customer_Age").alias("Age"),
            F.first("Churn").alias("Churn"),
            F.count("Customer_ID").alias("Number_Purchases")
        )
        self.df = self.df.withColumn('Total_Spent', self.df.Total_Price * self.df.Total_Quantity)
        self.df = self.df.withColumn('Gender', F.when(self.df.Sex == "Male", 0).otherwise(1))
        self.df = self.df.drop("Sex")

        self.cleaned_df = self.df.select("Customer_Name", "Total_Quantity", "Total_Price", 
                                         "Total_Spent", "Gender", 'Number_Purchases', "Age", "Churn")
        self.cleaned_df = self.cleaned_df.withColumnRenamed("Churn", "label")

    """
        Initalizing the VectorAssembler and Pipeline to create the logistic regression model.
        We'll be using total quantity, total price, gender, total spent, and age as our predictors.
        Splitting our dataset into a training and testing dataset using an 80/20 split.
    """
    def create_dataset(self) -> None:
        vector_assembler = VectorAssembler(
            inputCols=['Number_Purchases', 'Total_Price', 'Total_Quantity', 'Total_Spent', 'Age'], 
            outputCol='features'
        )
        customer_pipeline = Pipeline(stages=[vector_assembler])
        customer_piped_data = customer_pipeline.fit(self.cleaned_df).transform(self.cleaned_df)
        self.training, self.test = customer_piped_data.randomSplit([0.6, 0.4])
    
    def build_model(self) -> None:
        self.model = LogisticRegression()
        self.evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
        self.grid = tune.ParamGridBuilder()
        self.grid = self.grid.addGrid(self.model.regParam, np.arange(0, 0.1, 0.01))
        self.grid = self.grid.addGrid(self.model.elasticNetParam, [0, 1])
        self.grid = self.grid.build()
        self.cv = tune.CrossValidator(estimator=self.model, estimatorParamMaps=self.grid, evaluator=self.evaluator)
    
    def model_selection(self) -> None:
        models = self.cv.fit(self.training)
        self.best_model = models.bestModel

    def predict(self) -> None:
        self.results = self.best_model.transform(self.test)
        self.accuracy = self.evaluator.evaluate(self.results)
        print("Accuracy:", self.accuracy)

    def write_to_bigquery(self) -> None:
        try:
            df = self.cleaned_df.toPandas()
            pd.to_gbq(
                dataframe=df,
                destination_table=f"{project_id}.{dataset_id}.{table_id}",
                project_id=project_id,
                if_exists="replace" 
            )
            print("DataFrame loaded to BigQuery successfully.")
        except Exception as e:
            print("Failed to load DataFrame to BigQuery:", e)
            return
        print("Success!")
