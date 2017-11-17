fun main(args : Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 2,
            outputNeurons = 1,
            hiddenLayers = 2,
            neuronsPerHiddenLayer = 3
    )
    val flexNet = FlexNet(config)
    flexNet.print()

    println("\nPropagate!")
    flexNet.propagate(listOf(1.0))

    flexNet.print()
}