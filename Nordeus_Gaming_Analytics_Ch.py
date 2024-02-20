# Databricks notebook source
# MAGIC %md
# MAGIC #1: Does VALE achieve its goal?

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import Row, Column
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# define schemas

# Table 2
file_location_1 = "dbfs:/FileStore/tables/challenge_table_2_va.csv"

#schema 
schema_def_1 = StructType([StructField('date', DateType(), True),
                           StructField('global_user_id', DecimalType(), True),
                             StructField('time_utc', TimestampType(), True),
                             StructField('va_reward_reason', StringType(), True),
                             StructField('revenue_usd', DoubleType(), True)
])

# import data from DBFS
adv_df = spark.read.csv(file_location_1, encoding="UTF-8", header=True, schema=schema_def_1)

# COMMAND ----------

display(adv_df)

# COMMAND ----------

adv_df.groupBy(adv_df.va_reward_reason).agg(sum(col("revenue_usd")).alias("Total")).show(100)


# COMMAND ----------

adv_df_2 = adv_df.withColumn('va_reward_reason', regexp_replace('va_reward_reason','.*-.*', 'VALE'))
display(adv_df_2)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

# COMMAND ----------

adv_df_total = adv_df_2.groupBy(adv_df_2.va_reward_reason).agg(round(sum(col("revenue_usd")),2).alias("Total")).orderBy(desc("Total"))
adv_df_count = adv_df_2.groupBy(adv_df_2.va_reward_reason).agg(count(col("revenue_usd")).alias("Num")).orderBy(desc("Num"))
adv_df_total = adv_df_total.select("*").withColumn("Rank", monotonically_increasing_id())
adv_df_count = adv_df_count.select("*").withColumn("Rank", monotonically_increasing_id())


# COMMAND ----------

display(adv_df_total)

# COMMAND ----------

display(adv_df_count)

# COMMAND ----------

# define schemas

# Table 1
file_location_2 = "dbfs:/FileStore/tables/challenge_table_1_uw.csv"

#schema 
schema_def_2 = StructType([StructField('date', DateType(), True),
                           StructField('season', IntegerType(), True),
                           StructField('season_day', IntegerType(), True),
                           StructField('global_user_id', DecimalType(), True),
                           StructField('registration_date', DateType(), True),
                           StructField('cohort_day', IntegerType(), True),
                           StructField('registration_country_name', StringType(), True),
                           StructField('registration_platform', StringType(), True),
                           StructField('last_session_country', StringType(), True),
                           StructField('last_session_platform', StringType(), True),
                           StructField('session_count_daily', IntegerType(), True),
                           StructField('playtime_daily', DecimalType(), True),
                           StructField('training_count', DecimalType(), True),
                           StructField('iap_transaction_count_daily', IntegerType(), True),
                           StructField('net_revenue_usd_iap_daily', DoubleType(), True),
                           StructField('is_payer_lifetime', BooleanType(), True),
                           StructField('first_transaction_date_iap', DateType(), True),
                           StructField('tokens_earned', DecimalType(), True),
                           StructField('rest_boosters_earned', DecimalType(), True),
                           StructField('morale_boosters_earned', DecimalType(), True),
                           StructField('treatment_boosters_earned', DecimalType(), True),
                           StructField('rest_boosters_bought_for_tokens', DecimalType(), True),
                           StructField('morale_boosters_bought_for_tokens', DecimalType(), True),
                           StructField('treatment_boosters_bought_for_tokens', DecimalType(), True),
                           StructField('tokens_bought_by_iap', DecimalType(), True),
                           StructField('rest_boosters_bought_by_iap', DecimalType(), True),
                           StructField('morale_boosters_bought_by_iap', DecimalType(), True),
                           StructField('treatment_boosters_bought_by_iap', DecimalType(), True),
                           StructField('tokens_spent', DecimalType(), True),
                           StructField('rest_boosters_spent', DecimalType(), True),
                           StructField('morale_boosters_spent', DecimalType(), True),
                           StructField('treatment_boosters_spent', DecimalType(), True),
                           StructField('tokens_stash', IntegerType(), True),
                           StructField('rest_boosters_stash', IntegerType(), True),
                           StructField('morale_boosters_stash', IntegerType(), True),
                           StructField('treatment_boosters_stash', IntegerType(), True),
                           StructField('in_squad_auction_player_count', DecimalType(), True),
                           StructField('tokens_spent_on_auction_player', DecimalType(), True),
                           StructField('in_squad_scout_count', DecimalType(), True),
                           StructField('tokens_spent_on_scout', DecimalType(), True),
                           StructField('in_squad_daily_assistant_player_count', DecimalType(), True),
                           StructField('tokens_spent_on_daily_assistant_player', DecimalType(), True),
                           StructField('in_squad_recommended_player_count', DecimalType(), True),
                           StructField('tokens_spent_on_recommended_player', DecimalType(), True)
                           
])

# import data from DBFS
info_df = spark.read.csv(file_location_2, encoding="UTF-8", header=True, schema=schema_def_2)

# COMMAND ----------

print(info_df.count())
display(info_df)

# COMMAND ----------

agg_df = info_df.groupBy(info_df.date).agg(round(avg("playtime_daily")/60000).alias("avg_played_daily"), ceil(avg("session_count_daily")).alias("avg_login_daily"), count("date").alias("num_of_players"), round(sum("playtime_daily")/60000).alias("total_time_played"))

display(agg_df)

# COMMAND ----------

not_payers_df = info_df.select("global_user_id", "is_payer_lifetime","registration_country_name")
not_payers_df = not_payers_df.withColumnRenamed('global_user_id', 'global_user_id_t2')
not_payers_df = not_payers_df.dropDuplicates(["global_user_id_t2"])
not_payers_df = not_payers_df.sort("global_user_id_t2") 

display(not_payers_df)

not_payers_df.select(count(col("global_user_id_t2"))).show()
not_payers_removed_df = not_payers_df.where(col("is_payer_lifetime") == False)
display(not_payers_removed_df)

not_payers_removed_df.select(count(col("global_user_id_t2"))).show()
display(not_payers_df)

# COMMAND ----------

df_task_1 = info_df.select("global_user_id", "cohort_day", "playtime_daily", "date", "session_count_daily")
df_task_1 = df_task_1.groupBy("global_user_id").agg(round(avg("playtime_daily")/60000).alias("avg_played_daily"), max("cohort_day").alias("cohort_day"), count("date").alias("freq"), ceil(avg("session_count_daily")).alias("avg_login_daily"), round(sum("playtime_daily")/60000).alias("total_time_played"))
# print(df_task_1.filter(col("is_payer_lifetime") == False).count())
df_task_1 = df_task_1.join(not_payers_df, df_task_1.global_user_id == not_payers_df.global_user_id_t2, "inner")
# print(df_task_1.filter(col("is_payer_lifetime") == False).count())
df_task_1 = df_task_1.select("global_user_id", "avg_played_daily", "cohort_day","freq","avg_login_daily","is_payer_lifetime", "total_time_played")
df_task_1 = df_task_1.withColumnRenamed('global_user_id', 'global_user_id_t2')
df_task_1 = df_task_1.sort(["global_user_id_t2"])
display(df_task_1)
# print(df_task_1.count())

# COMMAND ----------

adv_df_3 = adv_df.withColumn('va_reward_reason', regexp_replace('va_reward_reason','.*-.*', 'VALE')).cache()
adv_df_3 = adv_df_3.withColumn('va_reward_reason', regexp_replace('va_reward_reason','^(?!VALE$).*$', 'NOT_VALE')).cache()

# COMMAND ----------

adv_df_4 = adv_df_3.join(not_payers_df, adv_df_3.global_user_id == not_payers_df.global_user_id_t2, "outer")
adv_df_4 = adv_df_4.withColumn('global_user_id',when(col('global_user_id').isNotNull(),col('global_user_id')).otherwise(col('global_user_id_t2')))
adv_df_4 = adv_df_4.na.fill(value="NONE",subset=["va_reward_reason"])
adv_df_4 = adv_df_4.na.fill(value=0,subset=["revenue_usd"])
adv_df_4 = adv_df_4.withColumn('va_reward_reason', regexp_replace('va_reward_reason','VALE', 'ADS'))
adv_df_4 = adv_df_4.withColumn('va_reward_reason', regexp_replace('va_reward_reason','NOT.*', 'ADS'))
# display(adv_df_4)

none_df = adv_df_4.select("global_user_id", "va_reward_reason").where(col("va_reward_reason") == "NONE")
none_df = none_df.dropDuplicates(["global_user_id"])
print(none_df.count())
ads_watch_df = adv_df_4.select("global_user_id", "va_reward_reason").where(col("va_reward_reason") != "NONE")
ads_watch_df = ads_watch_df.dropDuplicates(["global_user_id"])
print(ads_watch_df.count())
display(ads_watch_df)
display(adv_df_4)

# COMMAND ----------

by_player_df_0 = adv_df_3.select("global_user_id","va_reward_reason","revenue_usd").where(col("va_reward_reason") == 'VALE')
# print(by_player_df_0.count())
by_player_df_0 = by_player_df_0.groupBy(by_player_df_0.global_user_id, by_player_df_0.va_reward_reason).agg(count(col("revenue_usd")).alias("VALE")).orderBy(col("global_user_id")).cache()
# by_player_df_0.select(sum(col("VALE"))).show()
by_player_df_1 = adv_df_3.select("global_user_id","va_reward_reason","revenue_usd").where(col("va_reward_reason") == 'NOT_VALE')
# print(by_player_df_1.count())
by_player_df_1 = by_player_df_1.groupBy(by_player_df_1.global_user_id, by_player_df_1.va_reward_reason).agg(count(col("revenue_usd")).alias("NOT_VALE")).orderBy(col("global_user_id"))
by_player_df_1 = by_player_df_1.withColumnRenamed('global_user_id', 'global_user_id_2').cache()
# by_player_df_1.select(sum(col("NOT_VALE"))).show()
# by_player_df_0.show()
# by_player_df_1.show()

by_player_df_total = by_player_df_0.join(by_player_df_1, by_player_df_0.global_user_id == by_player_df_1.global_user_id_2, "outer")
by_player_df_total = by_player_df_total.withColumn('global_user_id',when(col('global_user_id').isNotNull(),col('global_user_id')).otherwise(col('global_user_id_2')))
by_player_df_total = by_player_df_total.select(by_player_df_total.global_user_id, by_player_df_total.VALE, col("NOT_VALE"))
by_player_df_total = by_player_df_total.na.fill(0)
by_player_df_total = by_player_df_total.sort('VALE','NOT_VALE').cache()
# by_player_df_total.select(sum(col("VALE")), sum(col("NOT_VALE"))).show()

print(by_player_df_total.count())
display(by_player_df_total)


# COMMAND ----------

new = by_player_df_total.join(not_payers_df, by_player_df_total.global_user_id == not_payers_df.global_user_id_t2, "inner")
display(new)
new_1 = new.select("global_user_id", "VALE", "NOT_VALE").where(col("is_payer_lifetime") == False)
new_2 = new.select("global_user_id", "VALE", "NOT_VALE").where(col("is_payer_lifetime") == True)
print(new_1.count())
print(new_2.count())

# COMMAND ----------

countries = info_df.select("registration_country_name")
countries = countries.dropDuplicates(["registration_country_name"])
display(countries)

# COMMAND ----------

country = by_player_df_total.join(not_payers_df, by_player_df_total.global_user_id == not_payers_df.global_user_id_t2, "inner")
country = country.filter(col("is_payer_lifetime") == False)
country_df = country.groupBy("registration_country_name").agg(count("global_user_id"), sum("VALE"), sum("NOT_VALE"))
display(country_df)

# COMMAND ----------

by_player_df_total.select(sum(by_player_df_total.VALE)).show()
by_player_df_total.select(sum(by_player_df_total.NOT_VALE)).show()


# COMMAND ----------

y = [val.VALE for val in new_1.select('VALE').collect()]
x = [val.NOT_VALE for val in new_1.select('NOT_VALE').collect()]

plt.plot(x, y, 'o', ms=5, color='#E6492A')

plt.ylabel('VALE')
plt.xlabel('NON-VALE')
plt.title('COUNT')

plt.show()

# COMMAND ----------

df_smaller = new_1.select("*").where(by_player_df_total.NOT_VALE <= 100)

y = [val.VALE for val in df_smaller.select('VALE').collect()]
x = [val.NOT_VALE for val in df_smaller.select('NOT_VALE').collect()]

plt.plot(x, y, 'o', ms=5, color='#E6492A')

plt.ylabel('VALE')
plt.xlabel('NON-VALE')
plt.title('COUNT')

plt.show()

# COMMAND ----------

from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# df_task_1
# by_player_df_total
# print(df_task_1.count())
df_pearson_2 = df_task_1.join(by_player_df_total, df_task_1.global_user_id_t2 == by_player_df_total.global_user_id, "outer")
df_pearson_2 = df_pearson_2.na.fill(0)

print(df_pearson_2.count())
ch = df_pearson_2.filter((col("VALE") == 0) & (col("NOT_VALE") == 0))
print(ch.count())
display(df_pearson_2)

# COMMAND ----------

from scipy.stats import pearsonr
from decimal import Decimal
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

corr_analysis_df_2 = df_pearson_2.select("avg_played_daily", "total_time_played", "avg_login_daily", "freq", "VALE", "NOT_VALE")
corr_analysis_df_2 = corr_analysis_df_2.select("avg_played_daily", "total_time_played", "avg_login_daily", "freq", "VALE", "NOT_VALE")
# display(corr_analysis_df_2)
# Convert PySpark DataFrame to pandas DataFrame
pandas_df = corr_analysis_df_2.toPandas()

# Convert decimal values to float
pandas_df = pandas_df.astype(float)

# Calculate the correlation coefficients and p-values
correlation_results = []
for col1 in pandas_df.columns:
    for col2 in pandas_df.columns:
        if col1 != col2:
            corr_coeff, p_value = pearsonr(pandas_df[col1], pandas_df[col2])
            correlation_results.append((col1, col2, corr_coeff, p_value))

# Create a DataFrame to store the results
df_corr_results = pd.DataFrame(correlation_results, columns=['Variable 1', 'Variable 2', 'Correlation Coefficient', 'p-value'])
# Print the results
display(df_corr_results)

# COMMAND ----------

from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sb

# COMMAND ----------

corr_analysis_df = df_pearson_2.select("avg_played_daily", "total_time_played", "avg_login_daily", "freq", "VALE", "NOT_VALE")
vector_col = "col-features"
assembler = VectorAssembler(inputCols=corr_analysis_df.columns, outputCol=vector_col)
df_vector = assembler.transform(corr_analysis_df).select(vector_col)
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corr_matrix = matrix.toArray().tolist()

colums = ['avg_played_daily', 'total_time_played', 'avg_login_daily', 'freq', 'VALE', 'NON_VALE']
# df_corr = spark.createDataFrame(corr_matrix, colums)
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = colums, index=colums) 
plt.figure(figsize=(16,5))  
sb.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True)
#corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)
#df_corr.show()

# COMMAND ----------

# MAGIC %md
# MAGIC avg_played_daily and VALE: The correlation coefficient is 0.366978, indicating a moderate positive correlation between average daily play and the VALE variable. <br>
# MAGIC The p-value is 0.0, indicating a highly significant correlation.
# MAGIC <br>
# MAGIC avg_played_daily and NOT_VALE: The correlation coefficient is 0.387831, indicating a moderate positive correlation between average daily play and the NOT_VALE variable. <br>
# MAGIC The p-value is 0.0, suggesting a highly significant correlation.
# MAGIC <br>
# MAGIC VALE and NOT_VALE: The correlation coefficient is 0.844952, indicating a strong positive correlation between the VALE and NOT_VALE variables. <br>
# MAGIC The p-value is 0.0, indicating a highly significant correlation.
# MAGIC <br>
# MAGIC freq and avg_played_daily: The correlation coefficient is 0.380088, indicating a moderate positive correlation between the freq and avg_played_daily variables. <br>
# MAGIC The p-value is 0.0, indicating a highly significant correlation.
# MAGIC <br>
# MAGIC freq and VALE: The correlation coefficient is 0.655908, indicating a moderate positive correlation between the freq and VALE variables. <br>
# MAGIC The p-value is 0.0, indicating a highly significant correlation.
# MAGIC <br>
# MAGIC NOT_VALE and freq: The correlation coefficient is 0.609135, indicating a moderate positive correlation between the NOT_VALE and freq variables. <br>
# MAGIC The p-value is 0.0, indicating a highly significant correlation.

# COMMAND ----------

task_2_df = adv_df.groupBy(adv_df.va_reward_reason).agg(count(col("revenue_usd")).alias("Total"))
task_2_df = task_2_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','ATTACK.*', 'VALE_ATTACK'))
task_2_df = task_2_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','DEFENCE.*', 'VALE_DEFENCE'))
task_2_df = task_2_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','MIDFIELD.*', 'VALE_MIDFIELD'))
task_2_df = task_2_df.groupBy("va_reward_reason").agg(sum(col("Total")).alias("Count"))
task_2_df = task_2_df.filter((col("va_reward_reason") == 'VALE_ATTACK') | (col("va_reward_reason") == 'VALE_DEFENCE') | (col("va_reward_reason") == 'VALE_MIDFIELD'))
display(task_2_df)
task_2_df.select(sum("Count")).show()

# COMMAND ----------

task_3_df = adv_df
task_3_df = task_3_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','ATTACK.*', 'VALE_ATTACK'))
task_3_df = task_3_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','DEFENCE.*', 'VALE_DEFENCE'))
task_3_df = task_3_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','MIDFIELD.*', 'VALE_MIDFIELD'))
task_3_df = task_3_df.filter((col("va_reward_reason") == 'VALE_ATTACK') | (col("va_reward_reason") == 'VALE_DEFENCE') | (col("va_reward_reason") == 'VALE_MIDFIELD'))

display(task_3_df)

# COMMAND ----------

ads_over_time_df = adv_df
ads_over_time_df = ads_over_time_df.withColumn('va_reward_reason', regexp_replace('va_reward_reason','^(?!.*-).*', 'NOT_VALE'))
ads_over_time_df = ads_over_time_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','ATTACK.*', 'ATTACK'))
ads_over_time_df = ads_over_time_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','DEFENCE.*', 'DEFENCE'))
ads_over_time_df = ads_over_time_df.withColumn("va_reward_reason", regexp_replace('va_reward_reason','MIDFIELD.*', 'MIDFIELD'))

ads_over_time_df = ads_over_time_df.groupBy("date", "va_reward_reason").agg(count("va_reward_reason").alias("NUMBER OF WATCHED ADS"))
display(ads_over_time_df.sort(["date", "va_reward_reason"]))
ads_over_time_df.filter(col("va_reward_reason") == "ATTACK").select(sum(col("NUMBER OF WATCHED ADS"))).show()
ads_over_time_df.filter(col("va_reward_reason") == "DEFENCE").select(sum(col("NUMBER OF WATCHED ADS"))).show()
ads_over_time_df.filter(col("va_reward_reason") == "MIDFIELD").select(sum(col("NUMBER OF WATCHED ADS"))).show()

# COMMAND ----------

total_over_time_df = adv_df_4.filter((col("va_reward_reason") == 'VALE')| (col("va_reward_reason") == 'NOT_VALE'))
total_over_time_df = total_over_time_df.groupBy("date", "va_reward_reason").agg(count("va_reward_reason").alias("NUMBER OF WATCHED ADS"))
display(total_over_time_df.sort(["date", "va_reward_reason"]))

# COMMAND ----------

train_over_time = info_df.groupBy("season", "season_day").agg(avg("session_count_daily").alias("avg_sessions"), avg("training_count").alias("avg_trainig_count"))
display(train_over_time)
train_over_time = info_df.groupBy("season", "season_day").agg(avg("session_count_daily"), avg("in_squad_auction_player_count"))
display(train_over_time)

# COMMAND ----------

auction_time_df = info_df.groupBy("date").agg(count("in_squad_auction_player_count"))
display(auction_time_df)

# COMMAND ----------

scout_df = info_df.groupBy("date").agg(count("in_squad_scout_count"))
display(scout_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Goal 3

# COMMAND ----------

adv_df_4_comb = adv_df_4.filter(col("is_payer_lifetime") == False)
adv_df_4_comb = adv_df_4_comb.select("global_user_id", "va_reward_reason")
adv_df_4_comb = adv_df_4_comb.dropDuplicates(["global_user_id", "va_reward_reason"])
print(adv_df_4_comb.count())
p = adv_df_4_comb.filter(col("va_reward_reason") == "ADS")
print(p.count())
p2 = adv_df_4_comb.filter(col("va_reward_reason") == "NONE")
print(p2.count())
display(adv_df_4_comb)

# COMMAND ----------

display(df_pearson_2)
df_pearson_2.select(sum(df_pearson_2.VALE)).show()
df_pearson_2.select(sum(df_pearson_2.NOT_VALE)).show()

# COMMAND ----------

goal_3_df = df_pearson_2.withColumn('WATCH', when((df_pearson_2.VALE > 0) | (df_pearson_2.NOT_VALE > 0), lit("ADS")).otherwise(lit("NONE")))
display(goal_3_df)

goal_3_df = goal_3_df.select("global_user_id_t2", "WATCH")
print(goal_3_df.count())
display(goal_3_df)

# COMMAND ----------

# display(info_df)

new_goal_3_df = info_df.join(goal_3_df, info_df.global_user_id == goal_3_df.global_user_id_t2, "outer")
print(new_goal_3_df.count())
display(new_goal_3_df)

# COMMAND ----------

first_tokens_earned = new_goal_3_df.groupBy("date","WATCH").agg(avg("tokens_earned"))
display(first_tokens_earned.sort(["date"]))

# COMMAND ----------

first_tokens_spent = new_goal_3_df.groupBy("date","WATCH").agg(avg("tokens_spent"))
display(first_tokens_spent.sort(["date"]))

# COMMAND ----------

first_tokens_stash = new_goal_3_df.groupBy("date","WATCH").agg(avg("tokens_stash"))
display(first_tokens_stash.sort(["date"]))

# COMMAND ----------

first_tokens_bought = new_goal_3_df.groupBy("date","WATCH").agg(avg("tokens_bought_by_iap"))
display(first_tokens_bought.sort(["date"]))

# COMMAND ----------

second_train = new_goal_3_df.groupBy("date","WATCH").agg(avg("training_count"))
display(second_train.sort(["date"]))

# COMMAND ----------



# COMMAND ----------

third_time = new_goal_3_df.groupBy("date","WATCH").agg(avg("session_count_daily"))
display(third_time.sort(["date"]))

# COMMAND ----------

third_time_2 = new_goal_3_df.groupBy("date","WATCH").agg(avg("playtime_daily")/60000)
display(third_time_2.sort(["date"]))

# COMMAND ----------

df_payers_check = new_goal_3_df.select("global_user_id", "is_payer_lifetime", "WATCH")
df_payers_check = df_payers_check.dropDuplicates(["global_user_id"])
display(df_payers_check.sort(["global_user_id"]))
df_payers_check = df_payers_check.filter((col("is_payer_lifetime") == True))
print(df_payers_check.count())

payers_ads = df_payers_check.filter((col("is_payer_lifetime") == True) & (col("WATCH") == "ADS"))
payers_ads = payers_ads.sort(["global_user_id"])
print(payers_ads.count())
display(payers_ads)

payers_ads_2 = df_payers_check.filter((col("is_payer_lifetime") == True) & (col("WATCH") == "NONE"))
payers_ads_2 = payers_ads_2.sort(["global_user_id"])
print(payers_ads_2.count())
display(payers_ads_2)
