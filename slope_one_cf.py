
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SQLContext
import operator
import math


# conf = SparkConf().setAppName("Slope One")
# sc = SparkContext(conf=conf)
# assert sc.version >= '1.5.1'
# sqlContext = SQLContext(sc)


def parse_lines(line):
    """
    Splits fields and turns user_id, item, rating to ints
    """
    fields = line.split(',')
    item = int(fields[2])
    user_id = int(fields[0])
    rating = int(fields[1])
    return (user_id, item, rating)

class SlopeOneCF():
    """Slope One Collaborative Filtering"""
    def __init__(self):
        conf = SparkConf().setAppName("Slope One")
        self.sc = SparkContext(conf=conf)
        assert self.sc.version >= '1.5.1'
        self.sqlContext = SQLContext(self.sc)
    
    def fit(self, filepath):
        lines = self.sc.textFile(filepath)
        header = lines.first() # extract header
        lines = lines.filter(lambda x: x != header)

        training_data = lines.map(parse_lines)

        training_RDD, val_RDD = training_data.randomSplit([8, 2], seed=0)
        print('Training Data Points', training_RDD.count())
        print('Validation Data Points', val_RDD.count())

        training_df = self.sqlContext.createDataFrame(training_RDD, ['uid', 'iid', 'rating'])
        testing_df = self.sqlContext.createDataFrame(val_RDD, ['uid', 'iid', 'rating'])

        training_df.registerTempTable("TrainingTable")
        testing_df.registerTempTable("TestingTable")

        # calculate the deviation between each item pairs. 
        # dev(j,i) = sum(r_j-r_i)/c(j,i)

        # all difference between ratings
        joined_user_df = self.sqlContext.sql("""
        SELECT t1.uid, t1.iid as iid1, t2.iid as iid2, (t1.rating - t2.rating) as rating_diff FROM
        TrainingTable t1
        JOIN
        TrainingTable t2
        ON (t1.uid = t2.uid)
        """)

        # sum / count of rating difference 
        # |iid1|iid2|                 dev|  c|
        joined_user_df.registerTempTable("JoinedUserTable")
        mpair_dev_c_df = self.sqlContext.sql("""
        SELECT iid1, iid2, sum(rating_diff) / count(rating_diff) as dev, count(rating_diff) as c FROM
        JoinedUserTable
        Group By iid1, iid2
        """)

        testing_training_df = self.sqlContext.sql("""
        SELECT t1.uid, t1.iid as iidj, t2.iid as iidi, t1.rating as rating_j, t2.rating as rating_i FROM
        TestingTable t1
        JOIN
        TrainingTable t2
        ON (t1.uid = t2.uid)
        """)

        # # join tables
        join_cond = [testing_training_df.iidj == mpair_dev_c_df.iid1, testing_training_df.iidi == mpair_dev_c_df.iid2]
        df = testing_training_df.join(mpair_dev_c_df, join_cond)

        # calculate how likely a user in the testing set will like an item. 
        # P(a,j) = sum((dev(j,i)+r(a,i))*c(j,i))/sum(c(j,i))
        df.registerTempTable("AllTable")
        ps = self.sqlContext.sql("""
        SELECT uid, iidj, sum((dev + rating_i) * c) / sum(c) as p, rating_j as true_rating FROM
        AllTable
        Group By uid, iidj, rating_j
        """)

        # calculate RMSE
        ps.registerTempTable("PTable")
        rmse = self.sqlContext.sql("""
        SELECT sqrt(sum(power(true_rating - p, 2))/count(true_rating)) as RMSE FROM
        PTable
        """)
        rmse.show()
        
        # calculate MAE
        ps.registerTempTable("PTable")
        mae = self.sqlContext.sql("""
        SELECT sum(abs(true_rating - p)) / count(true_rating) as MAE FROM
        PTable
        """)
        mae.show()
        
        self.sc.stop()
