package network

class FlexNetBenchmark (val k : Int, private val datasetPercentage: DatasetPercentage) {
    private val trainer = NetTrainer(MAX_TRIES = 1000)

    fun run() {
        var fileName = String()
        var config = FlexNetConfig(0, 0, 0, 0)
        var targetPosition = 0
        var hasId = false

        for (i in 1..4) {
            //choose file suffix based on data set percentage
            val fileSuffix = when (datasetPercentage) {
                DatasetPercentage.PCT100 -> "_100pct_fn.data"
                DatasetPercentage.PCT75 -> "_75pct_fn.data"
                DatasetPercentage.PCT50 -> "_50pct_fn.data"
                DatasetPercentage.PCT25 -> "_25pct_fn.data"
            }

            when (i) { //choose dataset and forest configuration
                1 -> {
                    fileName = "wine/wine$fileSuffix"
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
                    fileName = "haberman/haberman$fileSuffix"
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
                    fileName = "cmc/cmc$fileSuffix"
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
                    fileName = "wdbc/wdbc$fileSuffix"
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

        metricsList.forEach {
            meanAccuracy += it.accuracy
        }
        meanAccuracy /= metricsList.size

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

enum class DatasetPercentage {
    PCT100, PCT75, PCT50, PCT25
}

fun main(args : Array<String>) {
    val bench = FlexNetBenchmark(10, datasetPercentage = DatasetPercentage.PCT100)
    bench.run()
}