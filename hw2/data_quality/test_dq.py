import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from soda.scan import Scan


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
        .master("local") \
        .appName("chispa") \
        .getOrCreate()


def build_scan(name, spark_session):
    scan = Scan()
    scan.disable_telemetry()
    scan.set_scan_definition_name("data_quality_test")
    scan.set_data_source_name("spark_df")
    scan.add_spark_session(spark_session)
    return scan


def test_videos_source(spark):
    videos_df = spark.read.option('header', 'true').option(
        "inferSchema", "true").csv('datasets/USvideos.csv')
    videos_df.createOrReplaceTempView('videos')

    scan = build_scan("videos_source_data_quality_test", spark)
    scan.add_sodacl_yaml_file("data_quality/videos_checks.yml")

    scan.execute()

    scan.assert_no_checks_warn_or_fail()


def test_comments_source(spark):
    comments_all_df = spark.read.option('header', 'true')\
        .option('inferSchema', 'true').csv('datasets/UScomments.csv')
    comments_all_df.createOrReplaceTempView('comments_all')

    comments_schema = StructType([
        StructField('video_id', StringType(), True),
        StructField('comment_text', StringType(), True),
        StructField('likes', IntegerType(), True),
        StructField('replies', IntegerType(), True),
    ])

    comments_valid_df = spark.read.option('header', 'true')\
        .option('mode', 'DROPMALFORMED')\
        .schema(comments_schema)\
        .csv('datasets/UScomments.csv')
    comments_valid_df.createOrReplaceTempView('comments_valid')
    
    scan = build_scan("comments_source_data_quality_test", spark)
    scan.add_sodacl_yaml_file("data_quality/comments_checks.yml")

    scan.execute()

    scan.assert_no_checks_warn_or_fail()
