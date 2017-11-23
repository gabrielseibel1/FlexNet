class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, targetPosition : Int, hasId : Boolean) {

    private val dr = DataReader(dataFile, targetPosition, hasId)
    private val folding = Folding(dr.trainingDataSet, k)
    private val trainer = NetTrainer()
    private var bestConfigs = mutableListOf<Pair<FlexNetConfig, Metrics>>()
    private val MAX_BEST_CONFIGS_LIST_SIZE = 10

    fun doCrossValidation() {
        for (numberOfHiddenLayers in 1..4) {
            config.hiddenLayers = numberOfHiddenLayers
            for (numberOfNeurons in 1..4) {
                config.neuronsPerHiddenLayer = numberOfNeurons
                for(lambda in 1..5) {
                    config.lambda = lambda/1000.0
                    for(alpha in 1..10) {
                        config.alpha = alpha/100.0
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

                        //take means of metrics
                        val metrics  = calculateMetrics(flexNet)

                        println("Mean cost (J) = ${metrics.meanJ}")
                        println("Mean accuracy = ${metrics.meanAccuracy}")
                        println("Mean precision = ${metrics.meanPrecision}")
                        println("Mean recall = ${metrics.meanRecall}")
                        println("Stadard Deviation cost (J) = ${metrics.standardDeviationJ}")
                        println("Stadard Deviation accuracy = ${metrics.standardDeviationAccuracy}")
                        println("Stadard Deviation precision = ${metrics.standardDeviationPrecision}")
                        println("Stadard Deviation recall = ${metrics.standardDeviationRecall}")

                        saveIfGoodConfig(config.copy(), metrics)
                        //plot graph from data collected in training
                        //Plot(listOfTrainedFoldings.toDoubleArray(), listOfJs.toDoubleArray(), config.toString()).show()
                    }
                }
            }
        }
        println("\n\n!!!!!!!!!! BEST CONFIGS (${bestConfigs.size}) !!!!!!!!!!")
        println(bestConfigs)
    }

    private fun calculateMetrics(flexNet : FlexNet) : Metrics {
        var sumOfJs = 0.0
        var js = mutableListOf<Double>()
        var accuracies = mutableListOf<Double>()
        var precisions = mutableListOf<Double>()
        var recalls = mutableListOf<Double>()
        var sumOfAccuracies = 0.0
        var sumOfPrecisions = 0.0
        var sumOfRecalls = 0.0

        for (testFold in 0 until k) {
            js.add(flexNet.calculateJ(folding, testFold))

            sumOfJs += js[testFold]
            trainer.calculateConfusionMatrix(flexNet, folding.folds[testFold])


            accuracies.add(trainer.getAccuracy(flexNet))
            precisions.add(trainer.getPrecision(flexNet))
            recalls.add(trainer.getRecall(flexNet))

            sumOfAccuracies += trainer.getAccuracy(flexNet)
            sumOfPrecisions += trainer.getPrecision(flexNet)
            sumOfRecalls += trainer.getRecall(flexNet)
        }

        val meanJ = sumOfJs/k
        val meanAccuracy = sumOfAccuracies/k
        val meanPrecision = sumOfPrecisions/k
        val meanRecall = sumOfRecalls/k

        var standardDeviationJ = 0.0
        var standardDeviationAccuracy = 0.0
        var standardDeviationPrecision = 0.0
        var standardDeviationRecall = 0.0

        for(i in 0 until js.count()) {
            standardDeviationJ += Math.pow((js[i]-meanJ), 2.0)
            standardDeviationAccuracy += Math.pow((accuracies[i]-meanAccuracy), 2.0)
            standardDeviationPrecision += Math.pow((precisions[i]-meanPrecision), 2.0)
            standardDeviationRecall += Math.pow((recalls[i]-meanRecall), 2.0)
        }

        standardDeviationJ /= (k-1)
        standardDeviationAccuracy /= (k-1)
        standardDeviationPrecision /= (k-1)
        standardDeviationRecall /= (k-1)

        standardDeviationJ = Math.sqrt(standardDeviationJ)
        standardDeviationAccuracy = Math.sqrt(standardDeviationAccuracy)
        standardDeviationPrecision = Math.sqrt(standardDeviationPrecision)
        standardDeviationRecall = Math.sqrt(standardDeviationRecall)

        return Metrics(
                meanJ,
                meanAccuracy,
                meanPrecision,
                meanRecall,
                standardDeviationJ,
                standardDeviationAccuracy,
                standardDeviationPrecision,
                standardDeviationRecall
        )
    }

    fun saveIfGoodConfig(config: FlexNetConfig, metrics: Metrics) {

        if (bestConfigs.size < MAX_BEST_CONFIGS_LIST_SIZE) { //list of best configs has space
            bestConfigs.add(Pair(config, metrics))

        } else { //list is full

            //find worst config of the best configs
            var worstConfigIndex = 0
            bestConfigs.forEachIndexed { index, pair ->
                if (pair.second.meanAccuracy < bestConfigs[worstConfigIndex].second.meanAccuracy)
                    worstConfigIndex = index
            }

            //evaluate if should replace worst config for new one
            if (metrics.meanAccuracy > bestConfigs[worstConfigIndex].second.meanAccuracy)
                bestConfigs[worstConfigIndex] = Pair(config, metrics)
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