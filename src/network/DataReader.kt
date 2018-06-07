package network

import java.io.File
import java.util.*

class DataReader (val file : String, targetPosition : Int, hasId : Boolean) {

    private val dataString : String = File(file).readText()
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
}

fun main(args : Array<String>) {
    val dataReader : DataReader = DataReader("./data/haberman.data", 3, true)
    println(dataReader)
}