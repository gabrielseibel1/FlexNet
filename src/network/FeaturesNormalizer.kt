package network

class FeaturesNormalizer (private val targetPosition : Int) {

    private val targetAttributesKnown : MapOfTargetAttributes = MapOfTargetAttributes(targetAttributesKnown = mutableMapOf())

    fun normalizeFeatures(dataSet : MutableList<List<String>>) : Collection<Instance> {

        var auxDataSet : MutableList<Instance> = mutableListOf()
        val dataSetNormalized : MutableList<Instance> = mutableListOf()

        auxDataSet = convertToMutableInstance(dataSet)

        for(i in 1..auxDataSet.count()) {
            dataSetNormalized.add(Instance(
                    attributes = mutableListOf(),
                    targetAttribute = "",
                    targetAttributeNeuron = 0
            ))
        }

        for(feature in 1..auxDataSet[0].attributes.count()) {

            var maxValue : Double = Double.MIN_VALUE
            var minValue : Double = Double.MAX_VALUE

            auxDataSet.forEach {
                instance -> run {
                    if(instance.attributes[feature-1] > maxValue) {
                        maxValue = instance.attributes[feature-1]
                    }
                    if(instance.attributes[feature-1] < minValue) {
                        minValue = instance.attributes[feature-1]
                    }
                }
            }

            auxDataSet.forEachIndexed {
                index, instance -> run {
                    //println(instance.targetAttribute)
                    dataSetNormalized[index].targetAttributeNeuron = targetAttributesKnown.insertTargetAttribute(instance.targetAttribute)!!
                    dataSetNormalized[index].targetAttribute = instance.targetAttribute
                    dataSetNormalized[index].attributes.add((instance.attributes[feature-1]-minValue)/(maxValue-minValue))
                }
            }

        }

        return dataSetNormalized
    }

    private fun convertToMutableInstance(dataSet : MutableList<List<String>>) : MutableList<Instance> {

        val instanceConverted : MutableList<Instance> = mutableListOf()

        dataSet.forEach {
            data -> instanceConverted.add(convertToListDouble(data))
        }

        return instanceConverted
    }

    private fun convertToListDouble(instance : List<String>) : Instance {

        val instanceConverted = Instance(
                attributes = mutableListOf(),
                targetAttribute = "",
                targetAttributeNeuron = 0
        )

        instance.forEachIndexed {
            index, data -> run {
                if(index == targetPosition) instanceConverted.targetAttribute = data
                else instanceConverted.attributes.add(data.toDouble())
            }
        }

        return instanceConverted
    }
}

fun main(args : Array<String>) {
}