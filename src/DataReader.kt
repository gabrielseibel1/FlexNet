import java.io.File
import java.util.*

class DataReader (val file : String) {

    private val dataString : String = File(file).readText()
    private val dataSet : MutableList<List<String>> = mutableListOf()
    private val dataSetNormalized : MutableList<List<Double>> = mutableListOf()
    private val trainingDataSet : MutableList<List<Double>> = mutableListOf()
    private val testDataSet : MutableList<List<Double>> = mutableListOf()

    init {
        dataString.lines().forEach {
            line -> if(line != "") dataSet.add(line.split(","))
        }
        dataSetNormalized.addAll(FeaturesNormalizer.normalizeFeatures(dataSet))
        Collections.shuffle(dataSetNormalized)
        splitSetsToTrainAndTest()
    }

    private fun splitSetsToTrainAndTest() {
        val numberOfSets : Int = dataSetNormalized.count();
        val numberOfTrainingSets : Int = Math.floor(numberOfSets*0.8).toInt()
        for(i in 1..numberOfSets) {
            if(i-1 < numberOfTrainingSets) {
                trainingDataSet.add(dataSetNormalized[i-1])
            }
            else {
                testDataSet.add(dataSetNormalized[i-1])
            }
        }
    }

    fun getDataSet() : MutableList<List<String>> {
        return dataSet
    }

    fun getTrainingDataSet() : MutableList<List<Double>> {
        return trainingDataSet
    }

    fun getTestDataSet() : MutableList<List<Double>> {
        return testDataSet
    }
}

fun main(args : Array<String>) {
    val dataReader : DataReader = DataReader("./data/haberman.data")
}