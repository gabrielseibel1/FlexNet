class CrossValidation (val dataFile : String, val k : Int, val config : FlexNetConfig) {

    val dr = DataReader(dataFile)
    val folding = Folding(dr.getDataSet(), k)
    val flexNet = FlexNet(config)

    init {
    }

    fun doCrossValidation() {
        for (foldTest in 1..k) {
            for(foldPropagate in 1..k) {
                if(foldPropagate != foldTest) {
                    trainNet(folding.folds[foldPropagate-1])
                }
            }
        }
    }

    private fun trainNet(fold : MutableList<List<String>>) {
        for (instance in fold) {
            flexNet.propagate(convertToListDouble(instance))
            flexNet.print()
        }
    }

    private fun convertToListDouble(instance : List<String>) : List<Double> {
        val instanceConverted : MutableList<Double> = mutableListOf()
        instance.forEach {
            data -> instanceConverted.add(data.toDouble())
        }
        return instanceConverted.toList();
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 4,
            outputNeurons = 1,
            hiddenLayers = 2,
            neuronsPerHiddenLayer = 3
    )
    val cv = CrossValidation("./data/haberman.data", 10, config)
    cv.doCrossValidation()
}