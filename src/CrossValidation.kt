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
                    for(alpha in 6..6) {
                        config.alpha = alpha/10.0

                        //here we have a formed configuration to use an iterate over
                        val flexNet = FlexNet(config)
                        println("\n\n//////////////// CONFIG ////////////////")
                        println(config)

                        val listOfJs = mutableListOf<Double>()
                        val listOfTrainedFoldings = mutableListOf<Double>()
                        var trainedFoldings = 0

                        //repeat training until no more training is needed
                        var done = false
                        do {

                            for (testFold in 1..k) {
                                done = trainer.trainFolding(flexNet, folding.folds, testFold)

                                //add metrics to be plotted later
                                trainedFoldings++
                                listOfTrainedFoldings.add(trainedFoldings.toDouble())
                                listOfJs.add(flexNet.calculateJ(folding))

                                //if (done) break
                            }

                        } while (!done)

                        println("Trained foldings: $trainedFoldings")
                        println("${listOfTrainedFoldings.size} _ ${listOfJs.size}")

                        //now calculate metrics for current config

                        var sumOfJs = 0.0
                        var sumOfAccuracies = 0.0
                        var sumOfPrecisions = 0.0
                        var sumOfRecalls = 0.0

                        for (testFold in 1..k) {
                            sumOfJs += flexNet.calculateJFromFolds(folding.folds, testFold)
                            trainer.calculateConfusionMatrix(flexNet, folding.folds[testFold-1])
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
                        Plot(listOfTrainedFoldings.toDoubleArray(), listOfJs.toDoubleArray()).show()
                    }
                }
            }
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
    val cv = CrossValidation("./data/wdbc.data", 10, config, 0, true)
    cv.doCrossValidation()
    println("ok")
}