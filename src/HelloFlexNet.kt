fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 1,
            outputNeurons = 1,
            hiddenLayers = 2,
            neuronsPerHiddenLayer = 3,
            lambda = 0
    )
    val flexNet = FlexNet(config)
    flexNet.print()

    println("\nPropagate!")
    flexNet.propagate(listOf(0.5))

    flexNet.print()
}