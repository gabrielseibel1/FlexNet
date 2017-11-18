class CrossValidation (val dataFile : String, val k : Int, val config : FlexNetConfig, val classIsFirst : Boolean) {

    private val dr = DataReader(dataFile)
    private val folding = Folding(dr.getTrainingDataSet(), k)

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

    private fun trainNet(fold : MutableList<List<Double>>) {
        for (instance in fold) {
            val flexNet = FlexNet(config)
            if(classIsFirst) {
                flexNet.propagate(instance.subList(1, instance.count()))
                flexNet.print()
            }
            else {
                flexNet.propagate(instance.subList(0, instance.count()-1))
                flexNet.print()
            }
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 9,
            outputNeurons = 3,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3
    )
    val cv = CrossValidation("./data/cmc.data", 10, config, false)
    cv.doCrossValidation()
    println("ok")
}