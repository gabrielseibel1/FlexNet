fun main(args: Array<String>) {

    val config = FlexNetConfig(
            inputNeurons = 13,
            numberOfTargetAttributeClassesInDataSet = 3,
            hiddenLayers = 3,
            neuronsPerHiddenLayer = 4,
            lambda = 0.002,
            alpha = 0.01
    )

    val dr = DataReader("./data/wine.data", targetPosition = 0, hasId = false)
    //stop criteria checked every instance
    val trainer = NetTrainer(stepOfJCheck = 1, MIN_J_DIFFERENCE_PERCENTAGE = 0.00001, MAX_TRIES = 200000)

    val points = mutableListOf<Pair<Double, Double>>()
    var doneTraining = false
    val flexNet = FlexNet(config)
    //trains and collect J data until done
    do {
        doneTraining = trainer.trainAndCatalogJ(
                flexNet,
                trainingInstances = dr.trainingDataSet,
                testInstances = dr.testDataSet,
                points = points
        )
    } while (!doneTraining)

    println(points)
    Plot(points, config.toString()).show()
}