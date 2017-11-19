class CrossValidation (dataFile : String, val k : Int, config : FlexNetConfig, val targetPosition : Int) {

    private val dr = DataReader(dataFile, targetPosition)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val trainer = NetTrainer(FlexNet(config))

    fun doCrossValidation() {
        for (foldTest in 1..k) {
            for (foldPropagate in 1..k) {
                if(foldPropagate != foldTest) {
                    trainer.train(folding.folds[foldPropagate-1])
                }
            }
            trainer.flexNet.calculateJ(folding.folds, foldTest)
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 3,
            numberOfTargetAttributeClassesInDataSet = 2,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 4,
            lambda = 0.0
    )
    val cv = CrossValidation("./data/haberman.data", 10, config, 3)
    cv.doCrossValidation()
    println("ok")
}