from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import CountVectorizer
from cleantext import sanitize
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import DenseVector


def m_UDF(input):
    list = sanitize(input)
    return list[1].split() + list[2].split() + list[3].split()


def m_UDF2(original):
    return str(original[3:])


def m_UDF3(input):
	if input[1] > 0.2:
		return 1
	else:
		return 0

		
def m_UDF4(input):
	if input[1] > 0.25:
		return 1
	else:
		return 0


def main(sqlContext):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
	
	# TASK 1
	# Code for task 1…
    comments = sqlContext.read.json("comments-minimal.json.bz2")
    submissions = sqlContext.read.json("submissions.json.bz2")
    labeled_data = sqlContext.read.format('csv').options(header='true').load("labeled_data.csv")
    comments.write.parquet("comments.parquet")
    submissions.write.parquet("submissions.parquet")
    labeled_data.write.parquet("labeled_data.parquet")

	# test task 1
	# print(labeled_data)
    # Output: DataFrame[Input_id: string, labeldem: string, labelgop: string, labeldjt: string]
    # print(comments)
    # Output: DataFrame[author: string, author_cakeday: boolean, author_flair_css_class: string, author_flair_text: string,
    # body: string, can_gild: boolean, can_mod_post: boolean, collapsed: boolean, collapsed_reason: string,
    # controversiality: bigint, created_utc: bigint, distinguished: string, edited: string, gilded: bigint, id: string,
    # is_submitter: boolean, link_id: string, parent_id: string, permalink: string, retrieved_on: bigint, score: bigint,
    # stickied: boolean, subreddit: string, subreddit_id: string, subreddit_type: string]


	# TASK 2, 3
	# Code for tasks 2 and 3…
    # https://spark.apache.org/docs/preview/sql-programming-guide.html
    comments.createOrReplaceTempView("comments")
    labeled_data.createOrReplaceTempView("labeled_data")
    m_SQL = """select
        Input_id, labeldem, labelgop, labeldjt, body
        from labeled_data
        join comments
        on Input_id = id"""
		

	# TASK 4, 5
	# Code for tasks 4 and 5…
    # https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/sql/SQLContext.html
    new_table = sqlContext.sql(m_SQL)
    new_table.createOrReplaceTempView("new_table")
    # https://spark.apache.org/docs/1.1.0/api/python/pyspark.sql.SQLContext-class.html#registerFunction
    sqlContext.registerFunction("m_UDF", m_UDF, ArrayType(StringType()))
    m_SQL = """select
        Input_id, labeldem, labelgop, labeldjt, m_UDF(body) as parsed
        from new_table"""
    another_new_table = sqlContext.sql(m_SQL)
    another_new_table.createOrReplaceTempView("another_new_table")
    
	
	# test task 2, 3, 4, 5
	# m_SQL = """select
    #     Input_id, labeldem, labelgop, labeldjt, parsed
    #     from another_new_table
    #     limit 1"""
    # print(sqlContext.sql(m_SQL).take(1))
    # Output: [Row(Input_id='dhez0jx', labeldem='0', labelgop='0', labeldjt='1', parsed=['no', 'it', 'isnt', 'i', 'call',
    # 'to', 'all', 'fellow', 'proggresives', 'stop', 'with', 'that', 'bs', 'conspiracy', 'theory', 'and', 'help', 'trump',
    # 'lead', 'our', 'country', 'he', 'isnt', 'evil', 'or', 'anything', 'and', 'actually', 'is', 'improving', 'no_it',
    # 'it_isnt', 'i_call', 'call_to', 'to_all', 'all_fellow', 'fellow_proggresives', 'stop_with', 'with_that', 'that_bs',
    # 'bs_conspiracy', 'conspiracy_theory', 'theory_and', 'and_help', 'help_trump', 'trump_lead', 'lead_our', 'our_country',
    # 'he_isnt', 'isnt_evil', 'evil_or', 'or_anything', 'anything_and', 'and_actually', 'actually_is', 'is_improving', 'no_it_isnt',
    # 'i_call_to', 'call_to_all', 'to_all_fellow', 'all_fellow_proggresives', 'stop_with_that', 'with_that_bs', 'that_bs_conspiracy',
    # 'bs_conspiracy_theory', 'conspiracy_theory_and', 'theory_and_help', 'and_help_trump', 'help_trump_lead', 'trump_lead_our',
    # 'lead_our_country', 'he_isnt_evil', 'isnt_evil_or', 'evil_or_anything', 'or_anything_and', 'anything_and_actually',
    # 'and_actually_is', 'actually_is_improving'])]


	# TASK 6A
	# Code for task 6A
    vectorizing_function = CountVectorizer(inputCol="parsed", outputCol="results", minDF=10, binary=True)
	fitted = vectorizing_function.fit(another_new_table)
    ready = fitted.transform(another_new_table)
    ready.createOrReplaceTempView("ready")

	
	# test task 6A
	# print(ready.take(1))
	# Output: [Row(Input_id='dhez0jx', labeldem='0', labelgop='0', labeldjt='1', parsed=['no', 'it', 'isnt', 'i', 'call', 'to', 
	# 'all', 'fellow', 'proggresives', 'stop', 'with', 'that', 'bs', 'conspiracy', 'theory', 'and', 'help', 'trump', 'lead', 
	# 'our', 'country', 'he', 'isnt', 'evil', 'or', 'anything', 'and', 'actually', 'is', 'improving', 'no_it', 'it_isnt', 'i_call'
	# , 'call_to', 'to_all', 'all_fellow', 'fellow_proggresives', 'stop_with', 'with_that', 'that_bs', 'bs_conspiracy', 
	# 'conspiracy_theory', 'theory_and', 'and_help', 'help_trump', 'trump_lead', 'lead_our', 'our_country', 'he_isnt', 'isnt_evil'
	# , 'evil_or', 'or_anything', 'anything_and', 'and_actually', 'actually_is', 'is_improving', 'no_it_isnt', 'i_call_to', 
	# 'call_to_all', 'to_all_fellow', 'all_fellow_proggresives', 'stop_with_that', 'with_that_bs', 'that_bs_conspiracy', 
	# 'bs_conspiracy_theory', 'conspiracy_theory_and', 'theory_and_help', 'and_help_trump', 'help_trump_lead', 'trump_lead_our',
	# 'lead_our_country', 'he_isnt_evil', 'isnt_evil_or', 'evil_or_anything', 'or_anything_and', 'anything_and_actually',
	# 'and_actually_is', 'actually_is_improving'], results=SparseVector(1491, {1: 1.0, 3: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
	# 11: 1.0, 12: 1.0, 13: 1.0, 15: 1.0, 25: 1.0, 41: 1.0, 61: 1.0, 83: 1.0, 128: 1.0, 174: 1.0, 181: 1.0, 276: 1.0,
	# 408: 1.0, 417: 1.0, 552: 1.0, 672: 1.0, 1123: 1.0, 1227: 1.0, 1369: 1.0, 1409: 1.0, 1486: 1.0}))])]
	
	
	# TASK 6B
	# Code for tasks 6B
	# 6b  Create two new columns representing the positive and negative labels
	# first one is positive, second one is negative
	m_SQL = """SELECT *, 
	CASE 
		WHEN labeldjt = 1 
		then 1 
		ELSE 0 
		END 
	AS positive, 
	CASE 
		WHEN labeldjt = -1 
		then 1 
		ELSE 0 
		END 
		AS negative 
	FROM ready"""
	after = sqlContext.sql(m_SQL)
	after.createOrReplaceTempView("after")
	after.write.parquet("after.parquet")

	
	# TASK 7
	# Code for task 7
	m_SQL = """SELECT results AS features, positive AS label
	FROM after"""
	pos = sqlContext.sql(m_SQL)
	m_SQL = """SELECT results AS features, negative AS label
	FROM after"""
	neg = sqlContext.sql(m_SQL)
	# Initialize two logistic regression models.
	# Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
	poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
	neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
	# This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
	posEvaluator = BinaryClassificationEvaluator()
	negEvaluator = BinaryClassificationEvaluator()
	# There are a few parameters associated with logistic regression. We do not know what they are a priori.
	# We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
	# We will assume the parameter is 1.0. Grid search takes forever.
	posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
	negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
	# We initialize a 5 fold cross-validation pipeline.
	posCrossval = CrossValidator(
		estimator=poslr,
		evaluator=posEvaluator,
		estimatorParamMaps=posParamGrid,
		numFolds=5)
	negCrossval = CrossValidator(
		estimator=neglr,
		evaluator=negEvaluator,
		estimatorParamMaps=negParamGrid,
		numFolds=5)
	# Although crossvalidation creates its own train/test sets for
	# tuning, we still need a labeled test set, because it is not
	# accessible from the crossvalidator (argh!)
	# Split the data 50/50
	posTrain, posTest = pos.randomSplit([0.5, 0.5])
	negTrain, negTest = neg.randomSplit([0.5, 0.5])
	# Train the models
	print("Training positive classifier...")
	posModel = posCrossval.fit(posTrain)
	print("Training negative classifier...")
	negModel = negCrossval.fit(negTrain)
	# Once we train the models, we don't want to do it again. We can save the models and load them again later.
	posModel.save("project2/pos.model")
	negModel.save("project2/neg.model")


	# TASK 8
	# Code for task 8
	submissions.createOrReplaceTempView("submissions")
	sqlContext.registerFunction("m_UDF2", m_UDF2, StringType())
    m_SQL = """SELECT
		c.created_utc AS time_created,
		s.title AS title,
		c.score AS comments_score,
		s.score AS story_score,
		c.author_flair_text AS state,
		c.id AS comments_id,
		c.body AS comments_body
		FROM comments c
		JOIN submissions s
		ON m_UDF2(c.link_id) = s.id"""
    newer = sqlContext.sql(m_SQL)
	newer.createOrReplaceTempView("newer")
	newer.write.parquet("newer.parquet")


	# TASK 9
	# Code for task 9
	
	#take care of /s and qoute
	
	m_SQL = """SELECT * FROM newer WHERE
		comments_body NOT LIKE '%/s%' AND comments_body not like '&gt%'"""
	cleaned = sqlContext.sql(m_SQL)
	cleaned.createOrReplaceTempView("cleaned")
	m_SQL = """select
        time_created, title, state, comments_score, story_score, comments_id, m_UDF(comments_body) as parsed
        from cleaned"""
    another_cleaned = sqlContext.sql(m_SQL)
    another_cleaned.createOrReplaceTempView("another_cleaned")
    cleaned_ready = fitted.transform(another_cleaned)
    cleaned_ready.createOrReplaceTempView("cleaned_ready")
	cleaned_ready.write.parquet("cleaned_ready.parquet")
	
	#use the model to predict
	
	m_SQL = """SELECT time_created, title, state, story_score, comments_score, comments_id as id, results as features
		FROM cleaned_ready"""
	test_set = sqlContext.sql(m_SQL)
	pos_model = CrossValidatorModel.load("project2/pos.model")
    neg_model = CrossValidatorModel.load("project2/neg.model")
	positives = pos_model.transform(test_set)
	negatives = neg_model.transform(test_set)
	lambda_function = lambda threshold: udf(lambda vector: 1 if vector[1] > threshold else 0, IntegerType())
    positive_predicted = positives.select(lambda_function(0.2)("probability").alias("positive"), "id", "time_created", "title", "state", "comments_score", "story_score")
    negative_predicted = negatives.select(lambda_function(0.25)("probability").alias("negative"), "id", "time_created", "title", "state", "comments_score", "story_score")
	positive_predicted.createOrReplaceTempView("positive_predicted")
	negative_predicted.createOrReplaceTempView("negative_predicted")
	# positive_predicted.write.parquet("positive_predicted.parquet")
	# negative_predicted.write.parquet("negative_predicted.parquet")


	# TASK 10
	# Code for task 10
	avg_positive = positive_predicted.limit(2000000).agg(avg("positive")).collect()
	# [Row(avg(positive)=0.273246)]
	avg_negative = negative_predicted.limit(2000000).agg(avg("negative")).collect()
	# [Row(avg(negative)=0.641388)]
	
	sample = positive_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT state, avg(positive) as average_pos from sample group by state"""
	avg_positive_state = sqlContext.sql(m_SQL)
	avg_positive_state.write.format("com.databricks.spark.csv").option("header", "true").save("states.csv")
	
	sample = negative_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT state, avg(negative) as average_neg from sample group by state"""
	avg_negative_state = sqlContext.sql(m_SQL)
	avg_negative_state.write.format("com.databricks.spark.csv").option("header", "true").save("states_neg.csv")
	
	sample = positive_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT FROM_UNIXTIME(time_created, '%W, %Y%M%J'), avg(positive) as average_pos
	from sample group by FROM_UNIXTIME(time_created, '%W, %Y%M%J')"""
	avg_positive_time = sqlContext.sql(m_SQL)
	avg_positive_time.write.format("com.databricks.spark.csv").option("header", "true").save("time.csv")
	
	sample = negative_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT comments_score, avg(negative) as average_neg
	from sample group by comments_score"""
	avg_negative_cs = sqlContext.sql(m_SQL)
	avg_negative_cs.write.format("com.databricks.spark.csv").option("header", "true").save("cs_neg.csv")
	
	sample = positive_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT comments_score, avg(positive) as average_pos
	from sample group by comments_score"""
	avg_positive_cs = sqlContext.sql(m_SQL)
	avg_positive_cs.write.format("com.databricks.spark.csv").option("header", "true").save("cs.csv")
	
	sample = negative_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT story_score, avg(negative) as average_neg
	from sample group by story_score"""
	avg_negative_ss = sqlContext.sql(m_SQL)
	avg_negative_ss.write.format("com.databricks.spark.csv").option("header", "true").save("ss_neg.csv")
	
	sample = positive_predicted.limit(2000000)
	sample.createOrReplaceTempView("sample")
	m_SQL = """SELECT story_score, avg(positive) as average_pos
	from sample group by story_score"""
	avg_positive_ss = sqlContext.sql(m_SQL)
	avg_positive_ss.write.format("com.databricks.spark.csv").option("header", "true").save("ss.csv")


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)