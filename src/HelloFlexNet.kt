fun main(args : Array<String>) {
    val config = FlexNetConfig(
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3,
            inputNeurons = 1,
            outputNeurons = 1
    )
    val flexNet = FlexNet(config)
    flexNet.print()
}