class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, val targetPosition : Int) {

    private val dr = DataReader(dataFile, targetPosition)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val trainer = NetTrainer()

    fun doCrossValidation() {
        for (numberOfHiddenLayers in 1..1) {
            config.hiddenLayers = numberOfHiddenLayers
            for (numberOfNeurons in 2..2) {
                config.neuronsPerHiddenLayer = numberOfNeurons
                for(lambda in 12..12) {
                    config.lambda = lambda/10.0
                    for(alpha in 4..4) {
                        config.alpha = alpha/10.0
                        val flexNet = FlexNet(config)
                        for (foldTest in 1..k) {
                            for (foldPropagate in 1..k) {
                                if (foldPropagate != foldTest) {
                                    trainer.train(flexNet, folding.folds[foldPropagate - 1])
                                }
                            }
                            flexNet.calculateJ(folding.folds, foldTest)
                            trainer.calculateConfusionMatrix(flexNet, folding.folds[foldTest-1])
                            println(trainer.getAccuracy(flexNet))
                            println(trainer.getPrecision(flexNet))
                            println(trainer.getRecall(flexNet))
                        }
                    }
                }
            }
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 13,
            numberOfTargetAttributeClassesInDataSet = 3,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3,
            lambda = 0.0
    )
    val cv = CrossValidation("./data/wine.data", 10, config, 0)
    cv.doCrossValidation()
    println("ok")
}