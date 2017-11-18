class CrossValidation (val dataFile : String, val k : Int, val config : FlexNetConfig) {

    private val dr = DataReader(dataFile)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val flexNet = FlexNet(config)

    init {
    }

    fun doCrossValidation() {
        println(folding.folds)
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
            flexNet.propagate(instance)
        }
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