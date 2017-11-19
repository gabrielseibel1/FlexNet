class NetTrainer {

    fun train(flexNet: FlexNet, fold: Fold) {
        for (instance in fold.dataSet) {
            flexNet.forthAndBackPropagate(instance)
            flexNet.updateThetas()
        }
    }
}

fun main(args: Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 1,
            numberOfTargetAttributeClassesInDataSet = 2,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3
    )
    val flexNet = FlexNet(config)
    val netTrainer = NetTrainer()
    flexNet.print()
    val instance1 = Instance(mutableListOf(0.1/*,0.9*/), "Class1", 0)
    val instance2 = Instance(mutableListOf(0.2/*,0.8*/), "Class2", 1)
    val instances = listOf(instance1, instance2)
    netTrainer.train(flexNet, Fold(instances))
}