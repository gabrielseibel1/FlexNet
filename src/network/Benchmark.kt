package network

class Benchmark (val k : Int) {
    private val trainer = NetTrainer(MAX_TRIES = 1000)

    fun doCrossValidation() {
        var fileName = String()
        var config = FlexNetConfig(0, 0, 0, 0)
        var targetPosition = 0
        var hasId = false

        for (i in 1..4) {
            when (i) { //choose file and configuration
                1 -> {
                    fileName = "./data/wine_fn.data"//"wine_fn.data"
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
                    fileName = "./data/haberman_fn.data"//"haberman_fn.data"
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
                    fileName = "./data/cmc_fn.data"//"cmc_fn.data"
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
                    fileName = "./data/wdbc_fn.data"//"wdbc_fn.data"
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

            println("\n\n//////////////////// CONFIG ////////////////////")
            println(config)

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

            println("Mean cost (J) = ${meanMetrics.j}")
            println("Mean accuracy = ${meanMetrics.accuracy}")
        }
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
}

fun main(args : Array<String>) {
    val bench = Benchmark(10)
    bench.doCrossValidation()
    println("ok")
}