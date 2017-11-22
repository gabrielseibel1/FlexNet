class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, targetPosition : Int, hasId : Boolean) {

    private val dr = DataReader(dataFile, targetPosition, hasId)
    private val folding = Folding(dr.getTrainingDataSet(), k)
    private val trainer = NetTrainer()
    private var bestConfigs = mutableListOf<Pair<FlexNetConfig, Double>>()
    private val MAX_BEST_CONFIGS_LIST_SIZE = 10

    fun doCrossValidation() {
        for (numberOfHiddenLayers in 1..4) {
            config.hiddenLayers = numberOfHiddenLayers
            for (numberOfNeurons in 1..4) {
                config.neuronsPerHiddenLayer = numberOfNeurons
                for(lambda in 1..5) {
                    config.lambda = lambda/1000.0
                    for(alpha in 1..10) {
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


                        //now calculate final metrics for current config

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

                        saveIfGoodConfig(config.copy(), meanAccuracy)
                        //plot graph from data collected in training
                        //Plot(listOfTrainedFoldings.toDoubleArray(), listOfJs.toDoubleArray(), config.toString()).show()
                    }
                }
            }
        }
        println("\n\n!!!!!!!!!! BEST CONFIGS (${bestConfigs.size}) !!!!!!!!!!")
        println(bestConfigs)
    }

    fun saveIfGoodConfig(config: FlexNetConfig, accuracy: Double) {

        if (bestConfigs.size < MAX_BEST_CONFIGS_LIST_SIZE) { //list of best configs has space
            bestConfigs.add(Pair(config, accuracy))

        } else { //list is full

            //find worst config of the best configs
            var worstConfigIndex = 0
            bestConfigs.forEachIndexed { index, pair ->
                if (pair.second < bestConfigs[worstConfigIndex].second)
                    worstConfigIndex = index
            }

            //evaluate if should replace worst config for new one
            if (accuracy > bestConfigs[worstConfigIndex].second)
                bestConfigs[worstConfigIndex] = Pair(config, accuracy)
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 9,
            numberOfTargetAttributeClassesInDataSet = 3,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3,
            lambda = 0.00001
    )
    val cv = CrossValidation("./data/cmc.data", 10, config, 9, false)
    cv.doCrossValidation()
    println("ok")
}