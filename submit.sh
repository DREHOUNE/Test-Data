# Run on a Spark standalone cluster
spark-submit \
  --class domaine.KmeansApp \
  --master spark://ip:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 4G \
  --num-executors 3 \
  --total-executor-cores 4 \
  --conf spark.dynamicAllocation.enabled=true \
  Test-Data-1.0-SNAPSHOT.jar \
  1000


