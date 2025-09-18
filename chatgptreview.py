from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
import sys
import os
import urllib.request
import ssl

spark = SparkSession.builder.appName("LeaderboardOptimized").getOrCreate()

'''
# ==================== Sample Data ====================
cust_data = [
    (1, "Alice", "Hyderabad"),
    (2, "Bob", "Bangalore"),
    (3, "Charlie", "Chennai"),
    (4, "David", "Mumbai"),
    (5, "Eva", "Delhi")
]
cust_cols = ["cust_id", "name", "city"]

customers = spark.createDataFrame(cust_data, cust_cols)

txn_data = [
    (101, 1, "2023-08-01", 2500, "Card"),
    (102, 2, "2023-08-02", 4000, "UPI"),
    (103, 3, "2023-08-03", 1500, "Cash"),
    (104, 1, "2023-08-03", 3000, "Card"),
    (105, 4, "2023-08-04", 5000, "UPI"),
    (106, 2, "2023-08-05", 2000, "Card"),
    (107, 5, "2023-08-06", 7000, "UPI"),
    (108, 2, "2023-08-07", 8000, "Card"),
    (109, 1, "2023-08-08", 12000,"UPI"),
    (110, 3, "2023-08-09", 10000,"Cash")
]
txn_cols = ["txn_id","cust_id","txndate","amount","spendby"]

transactions = spark.createDataFrame(txn_data, txn_cols)

# ==================== Optimized Joins ====================
# Broadcast small dimension table (customers)
joindf = customers.join(transactions.hint("broadcast"), ["cust_id"], "left")

# ==================== Spend by method (skew safe) ====================
# Repartition by spendby to balance data
spendbydf = joindf.repartition("spendby") \
    .groupBy("name","spendby") \
    .agg(sum("amount").alias("each_customer"))

# ==================== 2nd highest transaction per customer ====================
wd = Window.partitionBy("cust_id").orderBy(col("amount").desc())
d_rank = joindf.withColumn("drank", dense_rank().over(wd))
second_highest_txn = d_rank.filter(col("drank") == 2).drop("drank")

# ==================== Max transaction per city ====================
max_txn_city = joindf.groupBy("city").agg(max("amount").alias("max_txn"))

# ==================== Customers who never used Cash ====================
no_cash = joindf.withColumn("cash_flag", when(col("spendby") == "Cash", 0).otherwise(1)) \
    .groupBy("cust_id","name") \
    .agg(sum("cash_flag").alias("non_cash_txns"),
         count("spendby").alias("total_txns")) \
    .filter(col("non_cash_txns") == col("total_txns")) \
    .select("cust_id","name")

# ==================== Leaderboard (Skew-safe aggregation) ====================
# 1. Add salt key for skew handling (spreads heavy customers)
salted = joindf.withColumn("salt", (rand()*10).cast("int"))

# 2. First aggregation with salt
agg1 = salted.groupBy("cust_id", "name", "salt").agg(
    sum("amount").alias("sum_part"),
    max("amount").alias("max_part"),
    count("amount").alias("count_part")
)

# 3. Merge salt partitions → final aggregation
agg2 = agg1.groupBy("cust_id", "name").agg(
    sum("sum_part").alias("total_amount"),
    max("max_part").alias("max_txn"),
    sum("count_part").alias("txn_count")
)

# 4. Compute avg_txn
agg2 = agg2.withColumn("avg_txn", (col("total_amount")/col("txn_count")))

# 5. Repartition on cust_id (for balance before sorting)
agg2 = agg2.repartition(100, "cust_id")

# 6. Top-N pattern (efficient)
leaderboard = agg2.orderBy(col("total_amount").desc()).limit(15)

# ==================== SHOW RESULTS ====================
print("========= Spend by Method =========")
spendbydf.show()

print("========= 2nd Highest Transaction =========")
second_highest_txn.show()

print("========= Max Transaction per City =========")
max_txn_city.show()

print("========= Customers with No Cash =========")
no_cash.show()

print("========= Leaderboard =========")
leaderboard.show()


data = [
    (101, "TV", 50000),
    (101, "Phone", 20000),
    (101, "Laptop", 70000),
    (102, "Phone", 15000),
    (102, "Tablet", 25000),
    (102, "Laptop", 40000),
]

columns = ["customer_id", "product", "amount"]

df = spark.createDataFrame(data, columns)
df.show()

wd= Window.partitionBy("customer_id").orderBy(col("amount").desc())
d_rank = df.withColumn("drank",dense_rank().over(wd))
d_rank.show()
results = d_rank.filter("drank<=2").drop("drank")
results.show()
'''

data = [
    ("U1", "2025-09-17 10:00:00", "login"),
    ("U1", "2025-09-17 10:05:00", "click"),
    ("U1", "2025-09-17 10:45:00", "click"),
    ("U1", "2025-09-17 11:10:00", "logout"),
    ("U2", "2025-09-17 09:55:00", "login"),
    ("U2", "2025-09-17 10:50:00", "logout"),
]

schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("action", StringType(), True),
])

df = spark.createDataFrame(data, schema)
df = df.withColumn("timestamp", df["timestamp"].cast(TimestampType()))
df.show(truncate=False)