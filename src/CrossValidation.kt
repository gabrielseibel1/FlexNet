class CrossValidation (dataFile : String, val k : Int, config : FlexNetConfig, val classIsFirst : Boolean) {

    private val dr = DataReader(dataFile)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val trainer = NetTrainer(FlexNet(config))

    fun doCrossValidation() {
        for (foldTest in 1..k) {
            for (foldPropagate in 1..k) {
                if(foldPropagate != foldTest) {
                    trainer.train(folding.folds[foldPropagate-1], classIsFirst)
                }
            }
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 9,
            outputNeurons = 3,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3,
            lambda = 0
    )
    val cv = CrossValidation("./data/cmc.data", 10, config, false)
    cv.doCrossValidation()
    println("ok")
}