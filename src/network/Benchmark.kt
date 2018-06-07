package network

class Benchmark (val k : Int) {
    private val trainer = NetTrainer(MAX_TRIES = 1000)

    fun run() {
        var fileName = String()
        var config = FlexNetConfig(0, 0, 0, 0)
        var targetPosition = 0
        var hasId = false

        for (i in 1..4) {
            when (i) { //choose file and configuration
                1 -> {
                    fileName = "wine_fn.data"
                    config = FlexNetConfig(
                            hiddenLayers = 1,
                            neuronsPerHiddenLayer = 4,
                            inputNeurons = 13,
                            numberOfTargetAttributeClassesInDataSet = 3,
                            lambda = 0.004,
                            alpha = 0.05
                    )
                    targetPosition = 0
                    hasId = false
                }
                2 -> {
                    fileName = "haberman_fn.data"
                    config = FlexNetConfig(
                            hiddenLayers = 2,
                            neuronsPerHiddenLayer = 4,
                            inputNeurons = 3,
                            numberOfTargetAttributeClassesInDataSet = 2,
                            lambda = 0.001,
                            alpha = 0.03
                    )
                    targetPosition = 3
                    hasId = false
                }
                3 -> {
                    fileName = "cmc_fn.data"
                    config = FlexNetConfig(
                            hiddenLayers = 1,
                            neuronsPerHiddenLayer = 4,
                            inputNeurons = 9,
                            numberOfTargetAttributeClassesInDataSet = 3,
                            lambda = 0.001,
                            alpha = 0.05
                    )
                    targetPosition = 9
                    hasId = false
                }
                4 -> {
                    fileName = "wdbc_fn.data"
                    config = FlexNetConfig(
                            hiddenLayers = 2,
                            neuronsPerHiddenLayer = 4,
                            inputNeurons = 30,
                            numberOfTargetAttributeClassesInDataSet = 2,
                            lambda = 0.001,
                            alpha = 0.08
                    )
                    targetPosition = 0
                    hasId = true
                }
            }

            val dr = DataReader(fileName, targetPosition, hasId)
            val folding = Folding(dr.trainingDataSet, k)

            val metricsList = mutableListOf<Metrics>()
            for (testFold in 0 until k) {
                trainer.resetTriesCounter()
                val flexNet = FlexNet(config)

                do {
                    val done = trainer.trainFolding(flexNet, folding, testFold)
                } while (!done)

                metricsList.add(calculateMetrics(flexNet, folding, testFold))
            }
            //take mean and std dev of metrics
            val meanMetrics = calculateMeanMetrics(metricsList.toList())


            println("$fileName { hl = ${config.hiddenLayers}, n/hl = ${config.neuronsPerHiddenLayer}, " +
                    "l = ${config.lambda}, a = ${config.alpha} } -> Acc = ${meanMetrics.accuracy}")
        }
    }

    private fun calculateMetrics(flexNet: FlexNet, folding: Folding, testFold: Int): Metrics {
        trainer.calculateConfusionMatrix(flexNet, folding.folds[testFold])
        return Metrics(
                0.0,
                trainer.getAccuracy(flexNet),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
        )
    }

    private fun calculateMeanMetrics(metricsList: List<Metrics>): Metrics {
        var meanAccuracy = 0.0
        /*var standardDeviationAccuracy = 0.0
        var meanJ = 0.0
        var meanPrecision = 0.0
        var meanRecall = 0.0
        var standardDeviationJ = 0.0
        var standardDeviationPrecision = 0.0
        var standardDeviationRecall = 0.0*/

        metricsList.forEach {
            //meanJ += it.j/metricsList.size
            meanAccuracy += it.accuracy
            //meanPrecision += it.precision/metricsList.size
            //meanRecall += it.recall/metricsList.size
        }
        meanAccuracy /= metricsList.size

        /*metricsList.forEach {
            standardDeviationAccuracy += Math.pow((it.accuracy-meanAccuracy), 2.0)/(metricsList.size-1)
            standardDeviationJ += Math.pow((it.j-meanJ), 2.0)/(metricsList.size-1)
            standardDeviationPrecision += Math.pow((it.precision-meanPrecision), 2.0)/(metricsList.size-1)
            standardDeviationRecall += Math.pow((it.recall-meanRecall), 2.0)/(metricsList.size-1)
        }*/

        /*standardDeviationAccuracy = Math.sqrt(standardDeviationAccuracy)
        standardDeviationJ = Math.sqrt(standardDeviationJ)
        standardDeviationPrecision = Math.sqrt(standardDeviationPrecision)
        standardDeviationRecall = Math.sqrt(standardDeviationRecall)*/

        return Metrics(
                0.0,
                meanAccuracy,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
        )
    }
}

fun main(args : Array<String>) {
    val bench = Benchmark(10)
    bench.run()
}