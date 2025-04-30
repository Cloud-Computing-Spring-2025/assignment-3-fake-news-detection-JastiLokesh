# Fake News Classification with Spark MLlib

This project implements a machine learning pipeline using Apache Spark MLlib to classify news articles as FAKE or REAL based on their content. The pipeline includes text preprocessing, feature extraction, model training, and evaluation.

## Dataset

The project uses the `fake_news_sample.csv` dataset, which contains news articles with the following columns:
- `id`: Unique identifier for each article
- `title`: The title of the news article
- `text`: The content/body of the news article
- `label`: The classification label (FAKE or REAL)

The dataset contains a sample of news articles that are labeled to indicate whether they are authentic (REAL) or fabricated (FAKE).

## Tasks Overview

### Task 1: Load & Basic Exploration
- Loads the CSV file with schema inference
- Creates a temporary view for SQL queries
- Performs basic exploration (counting articles, displaying samples, identifying labels)
- Outputs the initial DataFrame to task1_output.csv

### Task 2: Text Preprocessing
- Converts text to lowercase
- Tokenizes the text into words using Spark's Tokenizer
- Removes stopwords using StopWordsRemover
- Outputs the preprocessed data to task2_output.csv

### Task 3: Feature Extraction
- Implements TF-IDF vectorization:
  - HashingTF to calculate term frequencies
  - IDF to weigh down common terms across documents
- Converts categorical labels to numerical indices (FAKE → 0, REAL → 1)
- Outputs the feature vectors and labels to task3_output.csv

### Task 4: Model Training
- Splits data into training (80%) and test (20%) sets
- Trains a logistic regression model on the training data
- Generates predictions on the test data
- Outputs the predictions to task4_output.csv

### Task 5: Model Evaluation
- Evaluates the model using accuracy and F1 score metrics
- Outputs the evaluation results to task5_output.csv

## Implementation Details

The implementation uses several components from the Spark MLlib library:
- **Tokenizer**: Splits text into individual words
- **StopWordsRemover**: Removes common words that don't add meaning
- **HashingTF & IDF**: Implements TF-IDF feature extraction
- **StringIndexer**: Converts text labels to numerical indices
- **LogisticRegression**: Machine learning algorithm for classification
- **MulticlassClassificationEvaluator**: Tool for evaluating model performance

## How to Run the Code

### Prerequisites
- Apache Spark (2.4.x or later)
- Python 3.6+
- PySpark

### Execution Steps

1. Ensure the `fake_news_sample.csv` file is in your working directory.

2. Run the main script:
   ```
   spark-submit fake_news_classification.py
   ```

3. The script will generate the following output files:
   - `task1_output.csv`: Results from basic data exploration
   - `task2_output.csv`: Preprocessed text data
   - `task3_output.csv`: Feature vectors and indexed labels
   - `task4_output.csv`: Model predictions
   - `task5_output.csv`: Model evaluation metrics

### Code Structure

The code follows a sequential pipeline structure:
1. Initialize Spark session
2. Load and explore data (Task 1)
3. Preprocess text data (Task 2)
4. Extract features (Task 3)
5. Train and test model (Task 4)
6. Evaluate model performance (Task 5)

## Results

The logistic regression model achieves performance metrics (accuracy and F1 score) that are saved in task5_output.csv. These metrics indicate how well the model can distinguish between real and fake news articles based on their content.

## Future Improvements

Potential enhancements to this project could include:
- Experimenting with different feature extraction methods
- Trying other classification algorithms (Random Forest, Gradient Boosting)
- Implementing cross-validation for more robust model evaluation
- Enhancing text preprocessing with lemmatization or n-grams
- Adding feature importance analysis to understand key predictors of fake news