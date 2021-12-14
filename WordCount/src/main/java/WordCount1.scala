import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WordCount1 {

  def main(args : Array[String]) : Unit = {
    val sparConf = new SparkConf().setMaster("local").setAppName("WordCount")
    val sc = new SparkContext(sparConf)

    val lines: RDD[String] = sc.textFile("data")

    val words: RDD[String] = lines.flatMap(_.split(" "))

    val wordOne = words.map(
      word => (word, 1)
    )

    val group: RDD[(String, Iterable[(String, Int)])] = wordOne.groupBy(t => t._1)

    val wordCount = group.map {
      case (word, list) => {
        list.reduce(
          (t1, t2) => {
            (t1._1, t1._2 + t2._2)
          }
        )
      }
    }

    val array: Array[(String, Int)] = wordCount.collect()
    array.foreach(println)

    sc.stop()
  }
}
