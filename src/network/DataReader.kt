package network

import java.util.*

class DataReader (file : String, targetPosition : Int, hasId : Boolean) {

    private val inputStream = javaClass.classLoader.getResourceAsStream(file)
    private val dataString: String = convertStreamToString(inputStream)
    val dataSet : MutableList<List<String>> = mutableListOf()
    val dataSetNormalized : MutableList<Instance> = mutableListOf()
    val trainingDataSet : MutableList<Instance> = mutableListOf()
    val testDataSet : MutableList<Instance> = mutableListOf()

    init {
        val featuresNormalizer = FeaturesNormalizer(targetPosition)
        dataString.lines().forEach {
            line -> if(line != "") {
                var instance_aux = line.split(",")
                if(hasId) {
                    instance_aux = instance_aux.subList(1, instance_aux.count())
                }
                dataSet.add(instance_aux)
            }
            //println(line)
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

    private fun convertStreamToString(inputStream: java.io.InputStream): String {
        val s = java.util.Scanner(inputStream).useDelimiter("\\A")
        return if (s.hasNext()) s.next() else ""
    }
}

fun main(args : Array<String>) {
    val dataReader = DataReader("./data/haberman_fn.data", 3, true)
    println(dataReader)
}