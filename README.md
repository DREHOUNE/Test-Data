# Data-Test
In this test I will propose a code to perform a grouping based either on the location or on the characteristics of the bike stations, the data set is in the form of a json file, 

I will use K-means to build a machine learning model with Apache Spark and scala as the programming language. The K-Means model groups data according to station characteristics.  

# About K-Means 

K-Means clustering is one of the simplest and most popular unsupervised machine learning algorithms, I chose this model because it responds to our needs.  
The goal of this algorithm is to find clusters in the data, the number of clusters being represented by the variable K.  
The K-Means algorithm iteratively allocates each data point to the nearest cluster according to features. In each iteration of the algorithm, each data point is allocated to its nearest cluster according to a certain distance metric.  
The outputs of the algorithm are the centroids of the K clusters and the labels of the training data. Once the algorithm is running and has identified clusters from a data set, any new data can be easily assigned to a cluster. 

# Load data set 

To build K-Means model from this data set first we need to load this data set into spark DataFrame: 

```
val spark : SparkSession = SparkSession
        .builder()
        .appName("MyTestData")
        .master("local[1]")
        .getOrCreate()
        
// SparkSession has implicits
import spark.implicits._

 val data = spark.read.json("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike .json")
      data.show()
      data.printSchema()
```
The most suitable format for large volumes of data is Parquet in order to optimize Spark jobs, that's why I'm going to convert the Json file to Parquet:
```
 // Convert JSON to Parquet file
      data.write.parquet("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike.parquet")
      val parquetDF = spark.read.parquet("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike.parquet")
      parquetDF.show()

```
We need to transform features on the DataFrame records(latitude, longitude values) on each record into FeatureVector. In order to the features to be used by a machine learning algorithm this vector need to be added as a feature column into the DataFrame. Following is the way to do that with VectorAssembler: 

```
 // transform data with VectorAssembler to add feature column
      val cols = Array("latitude", "longitude")
      val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
      val featureDf = assembler.transform(parquetDF)

      featureDf.printSchema()
      featureDf.show()
```
Next, we can build the K-Means model by defining the characteristics column and the prediction column of the results. In order to train and test the K-Means model, the data set must be divided into a training data set and a test data set. 70% of the data is used for training the model and 30% will be used for testing. 

But first, we determine the optimal number of clusters for the data set, we compute Within Set Sum of Squared Error (WSSSE). You can reduce this error metric by increasing k. In fact, the optimal k is usually the one where there is a “elbow” in the WSSSE graph, and the result was k=10 

```
 // split data set training(70%) and test(30%)
      val seed = 1L
      val Array(trainingData, testData) = featureDf.randomSplit(Array(0.7, 0.3), seed)

      val k = Seq(2, 3, 4, 5, 6, 7, 8, 10,15, 20)
      val kmeans = new KMeans()
      val wssse = k.map(k => kmeans
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .fit(trainingData)
        .computeCost(testData)
      )
      println(wssse)

      // kmeans model with 10 clusters
      val model = kmeans.setK(10).setFeaturesCol("features")
        .setPredictionCol("prediction")
        .fit(trainingData)

      // test the model with test data
      val predictDf = model.transform(testData)
      predictDf.show()
 ```
 Save K-Means model 

The built model can be persisted in to disk in order to use later, stating the path: 
```
  // save model
      model.write.overwrite()
        .save("/Users/macbook/Desktop/Projects/Test-Data/results")

```
# continuous integration continuous delivery CI/CD: 

The first steps towards the industrialization of developments generally consist in the implementation of continuous integration. 

I used Jenkins for automation and job scheduling, I started by installing a Jenkins (locally) and configuring it ( download several plugins...) 
Jenkins relies on Jenkinsfile in which I defined the different tasks to be performed. 

Remark : I used the master to launch my jobs (in practice I have to create an agent and install it on the target machine) 

the first step is to make a "mvn clean install" which allows to build the jar. Jenkins by default creates a workspace in the target machine where it clones the project and executes the tasks defined in jenkinsfile (the path of this directory is retrieved from the variable ${env.WORKSPACE}). 

the next step is to run the spark job in the spark cluster using the spark-submit command that I put in a file called submit.sh that I created at the root of my project. below is the spark-submit command and the parameters that I used. 

I preferred to use dynamic allocation; in order to let Spark itself define the resources to allocate dynamically to each job.
Here, I will use the Spark standalone cluster manager, in order to configure an external shuffler on Spark standalone, start the worker with the key spark.shuffle.service.enabled set to true .
This can be done, for instance, through parameters to the spark-submit program, as follows:
```
# Run on a Spark standalone cluster
spark-submit \
  --class domain.KmeansApp \
  --master spark://ip:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 4G \
  --num-executors 3 \
  --total-executor-cores 4 \
  --conf spark.dynamicAllocation.enabled=true \
  Test-Data-1.0-SNAPSHOT.jar \
  1000
```
before launching spark-submit, I copied jar and also the submit.sh in the desired folder and put the path of this latter in a variable that I called MYWORKDIR. 

## Remarks: 
  1. I used java version 1.8
  2. Before executing the program you have to check the paths, even in the Jenkinsfile. 

 
