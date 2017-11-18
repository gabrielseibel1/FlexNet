class FeaturesNormalizer () {
    companion object {
        fun normalizeFeatures(dataSet : MutableList<List<String>>) : Collection<List<Double>> {
            var auxDataSet : MutableList<List<Double>> = mutableListOf()
            var dataSetNormalized : MutableList<MutableList<Double>> = mutableListOf()
            auxDataSet = convertToMutableListDouble(dataSet)
            for(i in 1..auxDataSet.count()) {
                dataSetNormalized.add(mutableListOf())
            }
            for(feature in 1..auxDataSet[0].count()) {
                var maxValue : Double = Double.MIN_VALUE
                var minValue : Double = Double.MAX_VALUE
                auxDataSet.forEach {
                    instance -> run {
                        if(instance[feature-1] > maxValue) {
                            maxValue = instance[feature-1]
                        }
                        if(instance[feature-1] < minValue) {
                            minValue = instance[feature-1]
                        }
                    }
                }
                auxDataSet.forEachIndexed {
                    index, instance -> run {
                        dataSetNormalized[index].add((instance[feature-1]-minValue)/(maxValue-minValue))
                    }
                }
            }
            return dataSetNormalized
        }

        fun convertToMutableListDouble(dataSet : MutableList<List<String>>) : MutableList<List<Double>> {
            val instanceConverted : MutableList<List<Double>> = mutableListOf()
            dataSet.forEach {
                data -> instanceConverted.add(convertToListDouble(data))
            }
            return instanceConverted;
        }

        fun convertToListDouble(instance : List<String>) : List<Double> {
            val instanceConverted : MutableList<Double> = mutableListOf()
            instance.forEach {
                data -> instanceConverted.add(data.toDouble())
            }
            return instanceConverted.toList();
        }
    }
}

fun main(args : Array<String>) {
}