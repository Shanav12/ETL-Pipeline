from pipeline import ETLPipeline
from login import database, port


etl = ETLPipeline(f"jdbc:mysql://localhost:{port}/{database}")
etl.table_to_dataframe('ecommerce_customers')
etl.group()
etl.write_to_bigquery()
etl.create_dataset()
etl.build_model()
etl.model_selection()
etl.predict()