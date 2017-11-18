import java.io.File

class DataReader (val file : String) {

    private val dataString : String = File(file).readText()
    private val dataSet : MutableList<List<String>> = mutableListOf()

    init {
        dataString.lines().forEach {
            line -> if(line != "") dataSet.add(line.split(","))
        }
    }



    fun getDataSet() : MutableList<List<String>> {
        return dataSet
    }
}

fun main(args : Array<String>) {
    val dataReader : DataReader = DataReader("./data/haberman.data")
}