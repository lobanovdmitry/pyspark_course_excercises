import pandas as pd
from pyspark.sql import functions as F, types
import pandas as pd
from bitarray.util import serialize
from video_analytics.bloom_filter import BloomFilter


def video_score(views: pd.Series, likes: pd.Series, dislikes: pd.Series, comment_likes: pd.Series, comment_replies: pd.Series) -> pd.Series:
    # some 'magic' score formula
    return views/10 + likes*100 - dislikes*10 + comment_likes*5 + comment_replies*10


video_score_udf = F.pandas_udf(video_score,
                               returnType=types.DoubleType())


def median(v: pd.Series) -> float:
    return v.median()


median_udf = F.pandas_udf(median,
                          returnType=types.DoubleType())


def split_tags(v: pd.Series) -> pd.Series:
    return v.apply(lambda tags: [tag for tag in tags.split('|') if tag])


split_tags_udf = F.pandas_udf(split_tags,
                              returnType=types.ArrayType(types.StringType()))

filterSize = 1000
prob = 0.1

def fill_bloom_filter(partition_id: int, df: pd.DataFrame) -> pd.DataFrame:
    bf = BloomFilter(filterSize, prob)
    for video_id in df['video_id']:
        bf.add(str(video_id))
    return pd.DataFrame({'part_id': partition_id, 'bf_array': serialize(bf.bit_array)})
