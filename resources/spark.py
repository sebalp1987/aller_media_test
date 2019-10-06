from abc import ABC, abstractmethod
from pyspark.sql import SparkSession


class SparkJob(ABC):
    """
    Interface that serves as a base for any Spark job
    The method run must be implemented by the child class
    Finally, the start method is used for starting the Job
    """

    @abstractmethod
    def run(self): raise NotImplementedError

    @staticmethod
    def get_spark_session(app_name="PMP Batch"):
        return SparkSession.builder\
            .master("local[*]")\
            .config('spark.executor.memory', '40g')\
            .config('spark.driver.memory', '40g')\
            .config("spark.sql.shuffle.partitions", 10)\
            .appName(app_name)\
            .getOrCreate()
