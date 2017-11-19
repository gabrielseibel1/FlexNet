class NetTrainer (private val flexNet: FlexNet) {

    fun train(fold: Fold) {
        for (instance in fold.dataSet) {
            println(instance)
            println("\nTraining...\n")
            flexNet.propagateAndBackPropagate(instance)
            flexNet.print()
        }
    }
}

fun main(args: Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 1,
            numberOfTargetAttributeClassesInDataSet = 2,
            hiddenLayers = 5,
            neuronsPerHiddenLayer = 3,
            lambda = 0
    )
    val flexNet = FlexNet(config)
    val netTrainer = NetTrainer(flexNet)
    flexNet.print()
    val instance1 = Instance(mutableListOf(0.1/*,0.9*/), "Class1", 0)
    val instance2 = Instance(mutableListOf(0.2/*,0.8*/), "Class2", 1)
    val instances = listOf(instance1, instance2)
    netTrainer.train(Fold(instances))
}