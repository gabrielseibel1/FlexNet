class FlexNet (val config : FlexNetConfig) {

    private val inputLayer : Layer = Layer(config.inputNeurons)
    private val hiddenLayers : List<Layer>
    private val outputLayer : Layer = Layer(config.outputNeurons)

    init {
        val init_layers = mutableListOf<Layer>()
        for (index in 1..config.hiddenLayers) {
            init_layers.add(Layer(config.neuronsPerHiddenLayer))
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