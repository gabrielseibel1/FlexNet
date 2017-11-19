class NetTrainer (private val flexNet: FlexNet) {

    fun train(fold: Fold) {
        for (instance in fold.instances) {
            println(instance)
            println("\nTraining...\n")
            flexNet.propagateAndBackPropagate(instance)
            flexNet.print()
        }
    }

    fun train(preFold: MutableList<List<Double>>, classIsFirstAttribute: Boolean) {
        val fold = Fold(preFold, classIsFirstAttribute)
        for (instance in fold.instances) {
            println(instance)
            println("\nTraining...\n")
            flexNet.propagateAndBackPropagate(instance)
            flexNet.print()
        }
    }
}

fun main(args: Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 2,
            outputNeurons = 1,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 2,
            lambda = 0
    )
    val netTrainer = NetTrainer(FlexNet(config))
    val instance1 = Instance(listOf(0.1,0.9), 1)
    val instance2 = Instance(listOf(0.2,0.8), 0)
    val instances = listOf(instance1, instance2)
    netTrainer.train(Fold(instances))
}