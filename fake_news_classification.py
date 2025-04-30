# Fake News Classification with Spark MLlib

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, lower, array_join

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Fake News Classification") \
    .getOrCreate()

# Task 1: Load & Basic Exploration
# --------------------------------

# Load the CSV file
news_df = spark.read.option("inferSchema", "true").option("header", "true").csv("fake_news_sample.csv")

# Create a temporary view
news_df.createOrReplaceTempView("news_data")

# Show the first 5 rows
print("First 5 rows:")
news_df.show(5)

# Count the total number of articles
article_count = news_df.count()
print(f"Total number of articles: {article_count}")

# Retrieve the distinct labels
distinct_labels = news_df.select("label").distinct().collect()
print("Distinct labels:")
for label in distinct_labels:
    print(label[0])

# Save the result to task1_output.csv
news_df.write.csv("task1_output.csv", header=True, mode="overwrite")

# Task 2: Text Preprocessing
# --------------------------

# Combine title and text for better feature extraction
news_df = news_df.withColumn("text", lower(col("text")))

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(news_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

# Create a temporary view for the cleaned data
cleaned_df.createOrReplaceTempView("cleaned_news")

# For saving to CSV, convert array column to string
# Use array_join to convert array to string with comma separator
cleaned_df_for_csv = cleaned_df.withColumn("filtered_words_str", array_join("filtered_words", ", "))

# Save the tokenized output to task2_output.csv
cleaned_df_for_csv.select("id", "title", "filtered_words_str", "label").write.csv("task2_output.csv", header=True, mode="overwrite")

# Task 3: Feature Extraction
# --------------------------

# TF-IDF Vectorization
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
featurized_df = hashingTF.transform(cleaned_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# Label Indexing
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed_df = label_indexer.fit(tfidf_df).transform(tfidf_df)

# For saving to CSV, convert array and vector columns to strings
indexed_df_for_csv = indexed_df.withColumn("filtered_words_str", array_join("filtered_words", ", "))
indexed_df_for_csv = indexed_df_for_csv.withColumn("features_str", col("features").cast("string"))

# Save the features and label indices to task3_output.csv
indexed_df_for_csv.select("id", "filtered_words_str", "features_str", "label_index").write.csv("task3_output.csv", header=True, mode="overwrite")

# Task 4: Model Training
# ---------------------

# Split the data into training and test sets
(training_data, test_data) = indexed_df.randomSplit([0.8, 0.2], seed=42)

# Train a logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)
lr_model = lr.fit(training_data)

# Generate predictions
predictions = lr_model.transform(test_data)

# Save the predictions to task4_output.csv
predictions.select("id", "title", "label_index", "prediction").write.csv("task4_output.csv", header=True, mode="overwrite")

# Task 5: Evaluate the Model
# -------------------------

# Evaluate the model using accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", 
    predictionCol="prediction", 
    metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Evaluate the model using F1 score
evaluator.setMetricName("f1")
f1 = evaluator.evaluate(predictions)

# Display metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Save metrics to task5_output.csv
metrics_df = spark.createDataFrame([
    ("Accuracy", float(accuracy)),
    ("F1 Score", float(f1))
], ["Metric", "Value"])

metrics_df.write.csv("task5_output.csv", header=True, mode="overwrite")

# Stop the Spark session
spark.stop()