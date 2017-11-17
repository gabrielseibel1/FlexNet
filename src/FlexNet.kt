class FlexNet (val config : FlexNetConfig) {

    private val inputLayer : Layer = Layer(config.inputNeurons, 0)
    private val outputLayer : Layer = Layer(config.outputNeurons, config.neuronsPerHiddenLayer)
    private val hiddenLayers : List<Layer>

    init {
        val init_layers = mutableListOf<Layer>()
        for (index in 1..config.hiddenLayers) {
            val neuronsThetaCount = if (index == 1) config.inputNeurons else config.neuronsPerHiddenLayer
            init_layers.add(Layer(config.neuronsPerHiddenLayer, neuronsThetaCount))
        }
        hiddenLayers= init_layers.toList()
    }

    fun print() {
        println(config)
        println("Input layer:")
        println(inputLayer)
        println("Hidden layers:")
        hiddenLayers.forEach { println(it) }
        println("Output layer:")
        println(outputLayer)
    }
}

fun main(args : Array<String>) {

}