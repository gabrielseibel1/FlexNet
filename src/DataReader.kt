import java.io.File
import java.util.*

class DataReader (val file : String, targetPosition : Int) {

    private val dataString : String = File(file).readText()
    private val dataSet : MutableList<List<String>> = mutableListOf()
    private val dataSetNormalized : MutableList<Instance> = mutableListOf()
    private val trainingDataSet : MutableList<Instance> = mutableListOf()
    private val testDataSet : MutableList<Instance> = mutableListOf()

    init {
        val featuresNormalizer = FeaturesNormalizer(targetPosition)
        dataString.lines().forEach {
            line -> if(line != "") dataSet.add(line.split(","))
        }
        dataSetNormalized.addAll(featuresNormalizer.normalizeFeatures(dataSet))
        Collections.shuffle(dataSetNormalized)
        splitSetsToTrainAndTest()
    }

    private fun splitSetsToTrainAndTest() {
        val numberOfSets : Int = dataSetNormalized.count()
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

    fun getTrainingDataSet() : MutableList<Instance> {
        return trainingDataSet
    }

    fun getTestDataSet() : MutableList<Instance> {
        return testDataSet
    }
}

fun main(args : Array<String>) {
    val dataReader : DataReader = DataReader("./data/haberman.data", 3)
}