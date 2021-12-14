import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WordCount2 {

  def main(args : Array[String]) : Unit = {
    val sparConf = new SparkConf().setMaster("local").setAppName("WordCount")
    val sc = new SparkContext(sparConf)

    var lines: RDD[String] = sc.textFile("data")

    lines = lines.map(_.toLowerCase())

    val words: RDD[String] = lines.flatMap(_.split(" "))

    val wordOne = words.map(
      word => (word, 1)
    )
    val wordCount = wordOne.reduceByKey(_+_)

    var array: Array[(String, Int)] = wordCount.collect()
    array = array.sortBy(_._2)
    array.foreach(println)

    sc.stop()
  }
}
