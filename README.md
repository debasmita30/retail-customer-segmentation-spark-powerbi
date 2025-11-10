# 🛍️ Retail Customer Segmentation using Apache Spark (Scala) and Power BI

[📊 View Dashboard](https://app.powerbi.com/view?r=eyJrIjoiZDdmZDg3NjEtZDBjYS00Mjk5LWFkNWMtZWZmYTI2MThiMjdhIiwidCI6ImUxNGU3M2ViLTUyNTEtNDM4OC04ZDY3LThmOWYyZTJkNWE0NiIsImMiOjEwfQ%3D%3D)

This project performs **customer segmentation for a retail dataset** using **Apache Spark (Scala)** and **K-Means clustering**, followed by visualization in **Power BI**.  
It helps identify **high-value customers**, analyze **buying behavior**, and design **data-driven marketing strategies**.

---

## 🚀 Project Overview

**Goal:** Cluster customers based on their purchasing behavior (Recency, Frequency, Monetary values)  
**Tech Stack:**  
- Apache Spark (3.4.1)  
- Scala (2.12.17)  
- Power BI (for dashboard visualization)  
- Kaggle Online Retail Dataset  

**Business Objective:**  
To segment customers into actionable groups — frequent buyers, occasional shoppers, and inactive customers — enabling targeted retention and marketing decisions.

---

## 🧩 Folder Structure

retail-customer-segmentation-spark-powerbi/
│
├── build.sbt # sbt configuration with Spark dependencies
├── src/
│ └── main/
│ └── scala/
│ └── CustomerSegmentationApp.scala # main Spark-Scala app
│
├── output/
│ └── CustomerClusters.csv # cleaned and clustered output data for Power BI
│
├── docs/
│ └── dashboard_screenshots/ # Power BI visuals (optional)
│
└── README.md # documentation

yaml
Copy code

---

## ⚙️ Prerequisites

Before running the project, install or configure:

- **Java 17+** and `JAVA_HOME`
- **Apache Spark 3.4+**
- **Scala 2.12**
- **sbt 1.11.7**
- **Power BI Desktop**
- Dataset: [Online Retail Dataset (Kaggle)](https://www.kaggle.com/datasets/)

---

## 🏗️ build.sbt

```scala
name := "CustomerSegmentation"

version := "0.1"

scalaVersion := "2.12.17"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.4.1",
  "org.apache.spark" %% "spark-sql" % "3.4.1",
  "org.apache.spark" %% "spark-mllib" % "3.4.1"
)
🧠 Scala Code — CustomerSegmentationApp.scala
scala
Copy code
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object CustomerSegmentationApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Customer Segmentation")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val dataPath = "C:/Users/Diya/Downloads/online+retail/OnlineRetail.csv"

    // Step 1: Load dataset
    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)

    // Step 2: Clean data
    val cleanedData = rawData
      .filter($"CustomerID".isNotNull)
      .filter($"Quantity" > 0)
      .filter($"UnitPrice" > 0)
      .withColumn("InvoiceAmount", $"Quantity" * $"UnitPrice")
      .withColumn("InvoiceDate", to_timestamp($"InvoiceDate", "dd-MM-yyyy HH:mm"))

    // Step 3: Compute RFM metrics
    val referenceDate = to_date(lit("2011-12-10"))
    val rfmData = cleanedData.groupBy("CustomerID")
      .agg(
        datediff(referenceDate, max(to_date($"InvoiceDate"))).alias("Recency"),
        countDistinct("InvoiceNo").alias("Frequency"),
        round(sum("InvoiceAmount"), 2).alias("Monetary")
      )

    // Step 4: Prepare data for K-Means
    val assembler = new VectorAssembler()
      .setInputCols(Array("Recency", "Frequency", "Monetary"))
      .setOutputCol("features")

    val featureData = assembler.transform(rfmData)

    // Step 5: Apply K-Means clustering
    val kmeans = new KMeans()
      .setK(10)
      .setSeed(42)
      .setFeaturesCol("features")
      .setPredictionCol("CustomerSegment")

    val model = kmeans.fit(featureData)
    val predictions = model.transform(featureData)

    // Step 6: Export for Power BI
    predictions
      .select("CustomerID", "Recency", "Frequency", "Monetary", "CustomerSegment")
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("C:/Users/Diya/Downloads/online+retail/output/CustomerClusters")

    println("✅ Export completed: CustomerClusters.csv generated successfully!")

    spark.stop()
  }
}
▶️ How to Run
Option 1: sbt Package + Spark Submit
bash
Copy code
cd C:\projects\retail-customer-segmentation-spark-powerbi
sbt package
%SPARK_HOME%\bin\spark-submit --class CustomerSegmentationApp --master local[*] target\scala-2.12\customersegmentation_2.12-0.1.jar
Option 2: Interactive spark-shell
bash
Copy code
%SPARK_HOME%\bin\spark-shell
Then paste in the commands from the Scala section.

📊 Power BI Dashboard
After export, load the CSV file into Power BI:

Dataset path:
C:/Users/Diya/Downloads/online+retail/output/CustomerClusters/CustomerClusters.csv

Visualizations created:
KPI Cards: Total Customers, Average Spending

Gauges: Average Days Since Last Purchase, Average Purchase Frequency

Scatter Plot: Spending vs. Frequency by Segment

Line & Clustered Column Chart: Total Spending & Avg Frequency by Segment

Stacked Bar Chart: Spending Breakdown by Customer Segment

Slicer: Filter by Total Spending (₹500 – ₹50,000)

Color-coded segments:

🟢 High-value frequent buyers

🟡 Moderately active customers

🔴 Inactive or lost customers

🧮 DAX Measures (Power BI)
DAX
Copy code
Total Customers = DISTINCTCOUNT(CustomerClusters[CustomerID])
Avg Recency = ROUND(AVERAGE(CustomerClusters[Recency]), 0)
Avg Frequency = ROUND(AVERAGE(CustomerClusters[Frequency]), 2)
Avg Monetary = ROUND(AVERAGE(CustomerClusters[Monetary]), 2)
Gauge color logic (optional 3-color version):

DAX
Copy code
GaugeColor_Frequency =
VAR CurrentValue = [Avg Frequency]
RETURN
    SWITCH(
        TRUE(),
        CurrentValue < 3, "#C00000",
        CurrentValue < 5, "#FFC000",
        "#00B050"
    )
🧾 Insights & Conclusions
The clustering revealed clear patterns in purchasing behavior, separating loyal, high-value customers from inactive ones.

Around 20% of customers generated a majority of revenue, indicating strong potential for targeted retention campaigns.

The Power BI dashboard offers an intuitive, visual understanding of customer engagement and spending patterns.

🧠 Learnings
How to process large datasets using Spark (Scala)

Implement K-Means clustering for segmentation

Build an interactive Power BI dashboard connected to Spark outputs

Use DAX to enhance visual interpretation
