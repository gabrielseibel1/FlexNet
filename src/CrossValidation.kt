class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, targetPosition : Int) {

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

                        //here we have a formed configuration to use an iterate over
                        val flexNet = FlexNet(config)
                        println(config)

                        var done = false

                        //train network one time with cross validation
                        for (testFold in 1..k) {
                            //add instances from training folds to one big list of instances called trainingInstances
                            val trainingFolds = folding.folds.filterIndexed{ index, _ -> index != testFold - 1 }
                            val trainingInstances = mutableListOf<Instance>()
                            trainingFolds.forEach { it.dataSet.forEach { trainingInstances.add(it) } }

                            trainer.trainBatch(flexNet, trainingInstances)
                        }

                        //repeat training until no more training is needed
                        trainer.resetTrainingTriesCount()
                        do {
                            for (testFold in 1..k) {
                                done = trainer.trainFolding(flexNet, folding.folds, testFold)
                                if (done) break
                            }
                        } while (!done)

                        //metrics for current config
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