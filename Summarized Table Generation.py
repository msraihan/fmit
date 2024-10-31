# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating DataFrame

# COMMAND ----------

from pyspark.sql import functions as F

query = """
SELECT * FROM (
    SELECT 
        row1, 
        row2, 
        row3,
        row4, 
        row5,
        row6,
        row7,
        row8
    FROM 'enter your directory'
    WHERE row1 in ('xxx','yyy')
    AND row4 like 'abcd'
    ) and row2 is not null
)
"""

df = spark.sql(query)
df_randomized = df.orderBy(F.rand()).limit(50)
display(df_randomized)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating JSON Summary

# COMMAND ----------

# creating json summary

%pip install openai
from openai import OpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType
import pandas as pd
import time

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = 'DB_TOKEN_HERE'
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def summarize_text(text, prompt):
    client = OpenAI(
      api_key=DATABRICKS_TOKEN,
      base_url="https://WORKSPACE_URL.net/serving-endpoints"
    )
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "you are a engine warranty administrator specializing in summarizing the failure information of a claim. You are only suppose to provide information that is seen on the warranty claim, nothing aside from that. If the information you are looking for is not available, leave it blank"
      },
      {
        "role": "user",
        "content": prompt + text
      }
      ],
      model="databricks-meta-llama-3-1-405b-instruct",
      max_tokens=4096
    )
    time.sleep(1.5)
    return chat_completion.choices[0].message.content

# Example DataFrame
df=df_randomized

# Collect data to the driver
data = df.select("correction", "cause", "complaint").collect()

# Customize the prompt
custom_prompt = "Summarize the following text, but I only care about the reason behind why the vehicle came in for service, not any other part return information. Just provide the summarized text, don't say 'here is a summary of the reason behind why the vehicle came in for service' or anything along those lines, just provide the JSON summary. Your goal is to understand and summarize why the vehicle came in for service, what was the cause, what was the correction. In your summary, include the fault codes found (they are 4 digit numbers). Provide the summary in a JSON format. The headings in the JSON should be: 'Reason for Service','Cause','Correction', and 'Fault Code' "

# Process data on the driver
summaries = [summarize_text(row['enter_row'], custom_prompt) for row in data]

# Create a new DataFrame with summaries
df_with_summaries = df.withColumn("enter_row", lit(None).cast(StringType()))

for i, summary in enumerate(summaries):
    df_with_summaries = df_with_summaries.withColumn("enter_row", 
                                                     when(col("enter_row") == data[i]['correction'], summary)
                                                     .otherwise(col("enter_row")))


df_with_summaries.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view") 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding JSON Summary

# COMMAND ----------

# embedding json summary

import logging
from openai import AzureOpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType

AZURE_OPENAI_KEY = "AZURE_KEY_HERE"
AZURE_OPENAI_ENDPOINT = "AZURE_ENDPOINT_HERE"
AZURE_OPENAI_API_VERSION = "AZURE_API_VERSION_HERE"
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = "AZURE_MODEL_HERE"

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT)

def create_embedding(text):
    response = azure_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_DEPLOYMENT_EMBEDDING
    )
    return [float(x) for x in response.data[0].embedding]

# Load the table
df = spark.table("table_name")

# Collect the summaries to the driver
data = df.select("summary_json").collect()

# Generate embeddings for each summary
embeddings = [create_embedding(row['summary_json']) for row in data]

# Add embeddings to the DataFrame
df_with_embeddings = df.withColumn("embedding_json", lit(None).cast(StringType()))

for i, embedding in enumerate(embeddings):
    df_with_embeddings = df_with_embeddings.withColumn("embedding_json", 
                                                       when(col("summary_json") == data[i]['summary_json'], str(embedding))
                                                       .otherwise(col("embedding_json")))

#display(df_with_embeddings)

df_with_embeddings.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Paragraph Summary from JSON Summary

# COMMAND ----------

# creating paragraph summary

%pip install openai
from openai import OpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType
import pandas as pd

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = 'DB_TOKEN_HERE'
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def summarize_text(text, prompt):
    client = OpenAI(
      api_key=DATABRICKS_TOKEN,
      base_url="https://WORKSPACE_URL.net/serving-endpoints"
    )
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "you are a engine warranty administrator specializing in summarizing the failure information of a claim. You are only suppose to provide information that is seen on the warranty claim, nothing aside from that. If the information you are looking for is not available, leave it blank."
      },
      {
        "role": "user",
        "content": prompt + text
      }
      ],
      model="databricks-meta-llama-3-1-405b-instruct",
      max_tokens=4000
    )
    time.sleep(1.5)
    return chat_completion.choices[0].message.content

# Example DataFrame
df = spark.table('table_name')

# Collect data to the driver
data = df.select("summary_json").collect()

# Customize the prompt
custom_prompt = "Summarize the following text, but I only care about the reason behind why the vehicle came in for service, not any other part return information. Just provide the summarized text, don't say 'here is a summary of the reason behind why the vehicle came in for service'. Your goal is to understand and summarize why the vehicle came in for service, what was the cause, what was the correction. In your summary, include the fault codes found (they are 4 digit numbers). Provide the text in a paragraph format "

# Process data on the driver
summaries = [summarize_text(row['summary_json'], custom_prompt) for row in data]

# Create a new DataFrame with summaries
df_with_summaries = df.withColumn("summary_paragraph", lit(None).cast(StringType()))

for i, summary in enumerate(summaries):
    df_with_summaries = df_with_summaries.withColumn("summary_paragraph", 
                                                     when(col("summary_json") == data[i]['summary_json'], summary)
                                                     .otherwise(col("summary_paragraph")))


df_with_summaries.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finding cause of failure

# COMMAND ----------

# creating paragraph summary

%pip install openai
from openai import OpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType
import pandas as pd

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = 'DB_TOKEN_HERE'
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def summarize_text(text, prompt):
    client = OpenAI(
      api_key=DATABRICKS_TOKEN,
      base_url="https://WORKSPACE_URL.net/serving-endpoints"
    )
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "you are a engine warranty administrator specializing in summarizing the failure information of a claim. You are only suppose to provide information that is seen on the warranty claim, nothing aside from that. If the information you are looking for is not available, leave it blank."
      },
      {
        "role": "user",
        "content": prompt + text
      }
      ],
      model="databricks-meta-llama-3-1-405b-instruct",
      max_tokens=4000
    )
    time.sleep(1.5)
    return chat_completion.choices[0].message.content

# Example DataFrame
df = spark.table('table_name')

# Collect data to the driver
data = df.select("summary_json").collect()

# Customize the prompt
custom_prompt = "Summarize the following text, but I only care about why the part failed . Just provide the summarized text, don't say 'here is a summary of the reason behind why the vehicle came in for service'. Your goal is to understand and summarize why the vehicle came in for service, what was the cause, what was the correction. In your summary, only include the reason why the part failed"

# Process data on the driver
summaries = [summarize_text(row['summary_json'], custom_prompt) for row in data]

# Create a new DataFrame with summaries
df_with_summaries = df.withColumn("llm_cause", lit(None).cast(StringType()))

for i, summary in enumerate(summaries):
    df_with_summaries = df_with_summaries.withColumn("llm_cause", 
                                                     when(col("summary_json") == data[i]['summary_json'], summary)
                                                     .otherwise(col("llm_cause")))


df_with_summaries.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding cause of failure

# COMMAND ----------

# embedding paragraph summary

import logging
from openai import AzureOpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType

AZURE_OPENAI_KEY = "AZURE_KEY_HERE"
AZURE_OPENAI_ENDPOINT = "AZURE_ENDPOINT_HERE"
AZURE_OPENAI_API_VERSION = "AZURE_API_VERSION_HERE"
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = "AZURE_MODEL_HERE"
AZURE_OPENAI_DEPLOYMENT_CHAT = "GPT_MODEL_HERE"

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT)

def create_embedding(text):
    response = azure_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_DEPLOYMENT_EMBEDDING
    )
    return [float(x) for x in response.data[0].embedding]

# Load the table
df = spark.table("table_name")

# Collect the summaries to the driver
data = df.select("llm_cause").collect()

# Generate embeddings for each summary
embeddings = [create_embedding(row['llm_cause']) for row in data]

# Add embeddings to the DataFrame
df_with_embeddings = df.withColumn("embedding_cause", lit(None).cast(StringType()))

for i, embedding in enumerate(embeddings):
    df_with_embeddings = df_with_embeddings.withColumn("embedding_cause", 
                                                       when(col("llm_cause") == data[i]['llm_cause'], str(embedding))
                                                       .otherwise(col("embedding_cause")))

#display(df_with_embeddings)

df_with_embeddings.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding paragraph summary

# COMMAND ----------

# embedding paragraph summary

import logging
from openai import AzureOpenAI
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType

AZURE_OPENAI_KEY = "AZURE_KEY_HERE"
AZURE_OPENAI_ENDPOINT = "AZURE_ENDPOINT_HERE"
AZURE_OPENAI_API_VERSION = "AZURE_API_VERSION_HERE"
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = "AZURE_MODEL_HERE"
AZURE_OPENAI_DEPLOYMENT_CHAT = "GPT_MODEL_HERE"

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT)

def create_embedding(text):
    response = azure_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_DEPLOYMENT_EMBEDDING
    )
    return [float(x) for x in response.data[0].embedding]

# Load the table
df = spark.table("table_name")

# Collect the summaries to the driver
data = df.select("summary_paragraph").collect()

# Generate embeddings for each summary
embeddings = [create_embedding(row['summary_paragraph']) for row in data]

# Add embeddings to the DataFrame
df_with_embeddings = df.withColumn("embedding_paragraph", lit(None).cast(StringType()))

for i, embedding in enumerate(embeddings):
    df_with_embeddings = df_with_embeddings.withColumn("embedding_paragraph", 
                                                       when(col("summary_paragraph") == data[i]['summary_paragraph'], str(embedding))
                                                       .otherwise(col("embedding_paragraph")))

#display(df_with_embeddings)

df_with_embeddings.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from table_name

# COMMAND ----------

# imports
import numpy as np
import pandas as pd
from ast import literal_eval

# load data
df_cluster = spark.sql('select * from table_name where rowname = "XXXX"').toPandas()
print(df_cluster)
df_cluster["embedding"] = df_cluster["embedding_cause"].apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df_cluster.embedding.values)
matrix.shape

# COMMAND ----------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Calculate distortions and silhouette scores for 1-10 clusters
distortions = []
silhouette_scores = []
K = range(1, 15)

for k in K:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    distortions.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(matrix, kmeans.labels_))
    else:
        silhouette_scores.append(float('nan'))  # Silhouette score is not defined for a single cluster

# Plot the elbow plot
plt.figure(figsize=(12, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method showing the optimal number of clusters')
plt.show()

# Plot the silhouette scores
plt.figure(figsize=(12, 6))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for different number of clusters')
plt.show()

# COMMAND ----------

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# Calculate the k-distance plot to choose a good eps value
k = 4  # or try k=5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(matrix)
distances, indices = neighbors_fit.kneighbors(matrix)

# Sort and plot distances to find the elbow
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.title("k-Distance Graph")
plt.xlabel("Data Points")
plt.ylabel("Distance to 4th Nearest Neighbor")
plt.show()


# COMMAND ----------

from sklearn.cluster import KMeans

n_clusters = 6

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
df_cluster["Cluster"] = labels

df_cluster.groupby("Cluster").failcode.count().sort_values()

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming matrix and df_cluster are already defined
tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42,
    init="random",
    learning_rate=200
)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

df_vis = pd.DataFrame({'x': x, 'y': y, 'Cluster': df_cluster['Cluster']})

colors = [
    "purple", "green", "red", "blue", "black", "yellow",
    "orange", "cyan", "magenta", "brown"
]
for category in range(n_clusters):
    color = colors[category % len(colors)]
    xs = np.array(x)[df_cluster.Cluster == category]
    ys = np.array(y)[df_cluster.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3, label=f"Cluster {category}")

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title("Clusters identified visualized in language 2d using t-SNE")
plt.legend()

# Convert Pandas DataFrame to Spark DataFrame
spark_df_vis = spark.createDataFrame(df_vis)

# Save the DataFrame to a table
spark_df_vis.createOrReplaceTempView("temp_view_vis")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_view_vis")

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tsne = TSNE(n_components=3, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims3 = tsne.fit_transform(matrix)

x = [x for x, y, z in vis_dims3]
y = [y for x, y, z in vis_dims3]
z = [z for x, y, z in vis_dims3]

colors = ["purple", "green", "red", "blue", "black", "yellow", "orange", "cyan", "magenta", "brown"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for category in range(n_clusters):
    color = colors[category % len(colors)]
    xs = np.array(x)[df_cluster.Cluster == category]
    ys = np.array(y)[df_cluster.Cluster == category]
    zs = np.array(z)[df_cluster.Cluster == category]
    ax.scatter(xs, ys, zs, color=color, alpha=0.3, label=f"Cluster {category}")

    avg_x = xs.mean()
    avg_y = ys.mean()
    avg_z = zs.mean()

    ax.scatter(avg_x, avg_y, avg_z, marker="x", color=color, s=100)
ax.set_title("Clusters identified visualized in language 3D using t-SNE")
ax.legend()

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

tsne = TSNE(n_components=3, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims3 = tsne.fit_transform(matrix)

df_vis = pd.DataFrame(vis_dims3, columns=['x', 'y', 'z'])
df_vis['Cluster'] = df_cluster['Cluster']

fig = px.scatter_3d(df_vis, x='x', y='y', z='z', color='Cluster', title="Clusters identified visualized in 3D using t-SNE")
fig.show()

# COMMAND ----------

df_cluster.columns

# COMMAND ----------

from openai import OpenAI
from openai import AzureOpenAI
import os

AZURE_OPENAI_KEY = "AZURE_KEY_HERE"
AZURE_OPENAI_ENDPOINT = "AZURE_ENDPOINT_HERE"
AZURE_OPENAI_API_VERSION = "AZURE_API_VERSION_HERE"
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = "AZURE_MODEL_HERE"
AZURE_OPENAI_DEPLOYMENT_CHAT = "GPT_MODEL_HERE"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Reading a review which belong to each group.
rev_per_cluster = 5
print_per_cluster = 3

cluster_responses = []

for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    reviews = "\n".join(
        df_cluster[df_cluster.Cluster == i]
        .summary_paragraph.str.replace("Title: ", "")
        .str.replace("\n\nContent: ", ":  ")
        .sample(rev_per_cluster, random_state=42)
        .values
    )

    messages = [
        {"role": "user", "content": f'You are a claims expert for engine warranties, specifically understanding engine turbocharger failures. Please summarize the common fail modes for the turbos between these technician notes and highlight the common fail points/mechanisms. You must pick the top ONE or most important failure mode that is common across all of the following samples: \n"""\n{reviews}\n"""\n\nTheme:'}
    ]

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_CHAT,
        messages=messages,
        max_tokens=4000,
        temperature=0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    
    theme = response.choices[0].message.content.replace("\n", "")
    print(theme)
    
    cluster_responses.append((i, theme))

    sample_cluster_rows = df_cluster[df_cluster.Cluster == i].sample(print_per_cluster, random_state=42)
    for j in range(print_per_cluster):
        print(sample_cluster_rows.failcode.values[j], end=", ")
        print(sample_cluster_rows.summary_json.values[j], end=":   ")

    print("-" * 100)

# Create a DataFrame from the cluster responses and write it to a table
df_cluster_responses = pd.DataFrame(cluster_responses, columns=['Cluster', 'Theme'])
spark_df_cluster_responses = spark.createDataFrame(df_cluster_responses)
# Save the DataFrame to a table
spark_df_cluster_responses.createOrReplaceTempView("spark_df_cluster_responses")
spark.sql("CREATE OR REPLACE TABLE table_name AS SELECT * FROM spark_df_cluster_responses")
