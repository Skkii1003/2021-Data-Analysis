import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args : Array[String]) : Unit = {

    //建立和spark框架的连接
    val sparConf = new SparkConf().setMaster("local").setAppName("WordCount")
    val sc = new SparkContext(sparConf)

    //执行业务操作

    //1.读取文件，获取每行数据
    val lines : RDD[String] = sc.textFile("data")

    //2.将每行数据拆分成单个单词（分词）
    // 扁平化：将整体拆分成个体
    val words : RDD[String] = lines.flatMap(_.split(" "))

    //3.将数据根据单词进行分组，便于统计
    val group : RDD[(String,Iterable[String])]= words.groupBy(word => word)

    //4.对分组后的数据进行转换
    val wordCount = group.map {
      case (word, list) => {
        (word, list.size)
      }
    }

    //5.将转换结果打印
    val array : Array[(String,Int)]= wordCount.collect()
    array.foreach(println)

    //关闭连接
    sc.stop()
  }
}
