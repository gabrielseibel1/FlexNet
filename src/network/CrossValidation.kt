package network

class CrossValidation (dataFile : String, val k : Int, val config : FlexNetConfig, targetPosition : Int, hasId : Boolean) {

    private val dr = DataReader(dataFile, targetPosition, hasId)
    private val folding = Folding(dr.trainingDataSet, k)
    private val trainer = NetTrainer(MAX_TRIES = 1000)
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

                        println("\n\n//////////////////// CONFIG ////////////////////")
                        println(config)

                        val metricsList = mutableListOf<Metrics>()
                        for (testFold in 0 until k) {
                            trainer.resetTriesCounter()
                            val flexNet = FlexNet(config)

                            var done = false
                            do {
                                done = trainer.trainFolding(flexNet, folding, testFold)
                            } while (!done)

                            metricsList.add(calculateMetrics(flexNet, folding, testFold))
                        }
                        //take mean and std dev of metrics
                        val meanMetrics = calculateMeanMetrics(metricsList.toList())

                        println("Mean cost (J) = ${meanMetrics.j}")
                        println("Mean accuracy = ${meanMetrics.accuracy}")
                        println("Mean precision = ${meanMetrics.precision}")
                        println("Mean recall = ${meanMetrics.recall}")
                        println("Stadard Deviation cost (J) = ${meanMetrics.standardDeviationJ}")
                        println("Stadard Deviation accuracy = ${meanMetrics.standardDeviationAccuracy}")
                        println("Stadard Deviation precision = ${meanMetrics.standardDeviationPrecision}")
                        println("Stadard Deviation recall = ${meanMetrics.standardDeviationRecall}")

                        saveIfGoodConfig(config.copy(), meanMetrics)
                    }
                }
            }
        }
        println("\n\n!!!!!!!!!! BEST CONFIGS (${bestConfigs.size}) !!!!!!!!!!")
        println(bestConfigs.joinToString(separator = "\n"))
    }

    private fun calculateMetrics(flexNet: FlexNet, folding: Folding, testFold: Int): Metrics {
        trainer.calculateConfusionMatrix(flexNet, folding.folds[testFold])
        return Metrics(
                flexNet.calculateJ(folding, testFold),
                trainer.getAccuracy(flexNet),
                trainer.getPrecision(flexNet),
                trainer.getRecall(flexNet),
                0.0,
                0.0,
                0.0,
                0.0
        )
    }

    private fun calculateMeanMetrics(metricsList: List<Metrics>): Metrics {
        var meanJ = 0.0
        var meanAccuracy = 0.0
        var meanPrecision = 0.0
        var meanRecall = 0.0
        var standardDeviationJ = 0.0
        var standardDeviationAccuracy = 0.0
        var standardDeviationPrecision = 0.0
        var standardDeviationRecall = 0.0

        metricsList.forEach {
            meanJ += it.j/metricsList.size
            meanAccuracy += it.accuracy/metricsList.size
            meanPrecision += it.precision/metricsList.size
            meanRecall += it.recall/metricsList.size
        }

        metricsList.forEach {
            standardDeviationJ += Math.pow((it.j-meanJ), 2.0)/(metricsList.size-1)
            standardDeviationAccuracy += Math.pow((it.accuracy-meanAccuracy), 2.0)/(metricsList.size-1)
            standardDeviationPrecision += Math.pow((it.precision-meanPrecision), 2.0)/(metricsList.size-1)
            standardDeviationRecall += Math.pow((it.recall-meanRecall), 2.0)/(metricsList.size-1)
        }

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

    private fun saveIfGoodConfig(config: FlexNetConfig, metrics: Metrics) {

        if (bestConfigs.size < MAX_BEST_CONFIGS_LIST_SIZE) { //list of best configs has space
            bestConfigs.add(Pair(config, metrics))

        } else { //list is full

            //find worst config of the best configs
            var worstConfigIndex = 0
            bestConfigs.forEachIndexed { index, pair ->
                if (pair.second.accuracy < bestConfigs[worstConfigIndex].second.accuracy)
                    worstConfigIndex = index
            }

            //evaluate if should replace worst config for new one
            if (metrics.accuracy > bestConfigs[worstConfigIndex].second.accuracy)
                bestConfigs[worstConfigIndex] = Pair(config, metrics)
        }
    }
}

fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 30,
            numberOfTargetAttributeClassesInDataSet = 2,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3,
            lambda = 0.00001
    )
    val cv = CrossValidation("./data/wdbc_fn.data", 10, config, 0, true)
    cv.doCrossValidation()
    println("ok")
}