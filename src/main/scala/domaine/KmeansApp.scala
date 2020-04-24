package domaine

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object KmeansApp {
    def main(args: Array[String]): Unit = {
      val spark: SparkSession = SparkSession
        .builder()
        .appName("MyTestData")
        .master("local[1]")
        .getOrCreate()

      // SparkSession has implicits
      import spark.implicits._

      val data = spark.read.json("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike .json")
      data.show()
      data.printSchema()

      // Convert JSON to Parquet file
      data.write.parquet("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike.parquet")
      val parquetDF = spark.read.parquet("/Users/macbook/Desktop/Projects/Test-Data/src/main/resources/Brisbane_CityBike.parquet")
      parquetDF.show()


      // transform data with VectorAssembler to add feature column
      val cols = Array("latitude", "longitude")
      val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
      val featureDf = assembler.transform(parquetDF)
      featureDf.printSchema()
      featureDf.show()

      // split data set training(70%) and test(30%)
      val seed = 1L
      val Array(trainingData, testData) = featureDf.randomSplit(Array(0.7, 0.3), seed)

      val k = Seq(2, 3, 4, 5, 6, 7, 8, 10, 15, 20)
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

      // calculate distance from center
      val distFromCenter = udf((features: Vector, c: Int) => Vectors.sqdist(features, model.clusterCenters(c)))
      val distanceDf = predictDf.withColumn("distance", distFromCenter($"features", $"prediction"))
      distanceDf.show(10)

      // no of categories
      predictDf.groupBy("prediction").count().show()

      // save model
      model.write.overwrite()
        .save("/Users/macbook/Desktop/Projects/Test-Data/results")

    }
}
