from chispa import *
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType
from video_analytics import functions as udf

# 1. Ваш авторитет в компании высок, но даже лучшие из нас иногда ошибаются.
# Поэтому вам нужно покрыть тестами ваши UDF-функции,
#  чтобы при будущих доработках не возникло проблем.

def test_video_score(spark: SparkSession):
    data = [
        (100, 10, 5, 100, 12, 1580.0),
        (10000, 0, 0, 0, 0, 1000.0),
        (0, 0, 0, 0, 0, 0.0),
        (100, 1, 1, 100, 1, 610.0),
    ]
    df = spark.createDataFrame(data, ['views', 'likes', 'dislikes', 'comment_likes', 'comment_replies', 'expected_score'])\
        .withColumn('score', udf.video_score_udf('views', 'likes', 'dislikes', 'comment_likes', 'comment_replies'))
    assert_column_equality(df, 'score', 'expected_score')


def test_median(spark: SparkSession):
    data = [x for x in range(1, 10)]
    rows = spark.createDataFrame(data, IntegerType())\
        .agg(udf.median_udf('value').alias('median'))\
        .collect()
    assert rows[0]['median'] == 5.0


def test_split_tags(spark: SparkSession):
    data = [
        ('cat|dog|food', ['cat', 'dog', 'food']),
        ('cat', ['cat']),
        ('|cat', ['cat']),
        ('cat|', ['cat']),
        ('|cat|', ['cat']),
    ]
    df = spark.createDataFrame(data, ['tags', 'expected_tags'])\
        .withColumn('splitted_tags', udf.split_tags_udf('tags'))
    assert_column_equality(df, 'splitted_tags', 'expected_tags')


def test_bloom_filter(spark: SparkSession):
    from bitarray.util import deserialize
    from video_analytics.bloom_filter import BloomFilter

    data = ['v'+str(x) for x in range(0, 999)]
    df = spark.createDataFrame(data, StringType())\
        .withColumnRenamed('value', 'video_id')\
        .groupBy(F.spark_partition_id())\
        .applyInPandas(udf.fill_bloom_filter, 'part_id long, bf_array array<long>')\
        .collect()

    bf_array = deserialize(bytes(df[0]['bf_array']))
    bf = BloomFilter(udf.filterSize, udf.prob)
    bf.set_bit_array(bf_array)
    for video_id in data:
        assert bf.check(video_id) != False
