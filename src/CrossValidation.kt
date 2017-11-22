class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, targetPosition : Int, hasId : Boolean) {

    private val dr = DataReader(dataFile, targetPosition, hasId)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val trainer = NetTrainer()

    fun doCrossValidation() {
        for (numberOfHiddenLayers in 1..1) {
            config.hiddenLayers = numberOfHiddenLayers
            for (numberOfNeurons in 2..2) {
                config.neuronsPerHiddenLayer = numberOfNeurons
                for(lambda in 1..1) {
                    config.lambda = lambda/10000000.0
                    for(alpha in 1..6) {
                        config.alpha = alpha/10.0

                        //here we have a formed configuration to use an iterate over
                        val flexNet = FlexNet(config)
                        println("\n\n//////////////// CONFIG ////////////////")
                        println(config)

                        val listOfJs = mutableListOf<Double>()
                        val listOfTrainedFoldings = mutableListOf<Double>()
                        var trainedFoldings = 0

                        //get initial J
                        flexNet.propagate(folding.dataSet[0].attributes)
                        listOfTrainedFoldings.add(trainedFoldings.toDouble())
                        listOfJs.add(flexNet.calculateJ(folding, 0))

                        //repeat training until no more training is needed
                        trainer.resetTriesCounter()
                        var done = false
                        do {

                            var meanJOfFoldings = 0.0
                            var foldingsCount = 0
                            for (testFold in 0 until k) {
                                done = trainer.trainFolding(flexNet, folding, testFold)

                                //add metrics to be plotted later
                                trainedFoldings++
                                meanJOfFoldings += flexNet.calculateJ(folding, testFold)

                                //counts how many foldings in this round we're trained to calculate mean J
                                foldingsCount++
                                if (done) break
                            }
                            meanJOfFoldings /= foldingsCount
                            listOfTrainedFoldings.add(trainedFoldings.toDouble())
                            listOfJs.add(meanJOfFoldings)

                        } while (!done)

                        println("Trained foldings: $trainedFoldings")
                        println("${listOfTrainedFoldings.size} _ ${listOfJs.size}")


                        //now calculate metrics for current config

                        var sumOfJs = 0.0
                        var sumOfAccuracies = 0.0
                        var sumOfPrecisions = 0.0
                        var sumOfRecalls = 0.0

                        for (testFold in 0 until k) {
                            sumOfJs += flexNet.calculateJ(folding, testFold)
                            trainer.calculateConfusionMatrix(flexNet, folding.folds[testFold])
                            sumOfAccuracies += trainer.getAccuracy(flexNet)
                            sumOfPrecisions += trainer.getPrecision(flexNet)
                            sumOfRecalls += trainer.getRecall(flexNet)
                        }

                        //take means of metrics
                        val meanJ = sumOfJs/k
                        val meanAccuracy = sumOfAccuracies/k
                        val meanPrecision = sumOfPrecisions/k
                        val meanRecall = sumOfRecalls/k

                        println("Mean cost (J) = $meanJ")
                        println("Mean accuracy = $meanAccuracy")
                        println("Mean precision = $meanPrecision")
                        println("Mean recall = $meanRecall")

                        //plot graph from data collected in training
                        Plot(listOfTrainedFoldings.toDoubleArray(), listOfJs.toDoubleArray(), config.toString()).show()
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
            lambda = 0.00001
    )
    val cv = CrossValidation("./data/wine.data", 10, config, 0, false)
    cv.doCrossValidation()
    println("ok")
}