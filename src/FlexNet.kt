class FlexNet (val config : FlexNetConfig) {

    private val inputLayer : Layer = Layer(config.inputNeurons, 1)
    private val outputLayer : Layer = Layer(config.numberOfTargetAttributeClassesInDataSet, config.neuronsPerHiddenLayer)
    private val hiddenLayers : List<Layer>

    init {
        val initLayers = mutableListOf<Layer>()
        for (index in 1..config.hiddenLayers) {
            val neuronsThetaCount = if (index == 1) config.inputNeurons else config.neuronsPerHiddenLayer
            initLayers.add(Layer(config.neuronsPerHiddenLayer, neuronsThetaCount))
        }
        hiddenLayers= initLayers.toList()
    }

    fun propagateAndBackPropagate(instance: Instance) {
        propagate(instance.attributes)
        backPropagate(instance.targetAttributeNeuron)
    }

    fun propagate(inputs: List<Double>) {
        activateInputLayer(inputs)
        var previousLayer = inputLayer
        hiddenLayers.forEach {
            it.activate(previousLayer)
            previousLayer = it
        }
        outputLayer.activate(hiddenLayers.last())
    }

    private fun backPropagate(correctOutput: Int) {
        val correctOutputs = buildCorrectOutputs(correctOutput)
        println("Correct outputs $correctOutputs")
        var previousLayer = outputLayer
        outputLayer.calculateDeltasFromCorrectOutputs(correctOutputs)
        hiddenLayers.asReversed().forEach {
            it.calculateDeltas(previousLayer)
            previousLayer = it
        }
    }

    private fun buildCorrectOutputs(correctOutput: Int) : List<Int> {
        val correctOutputs = mutableListOf<Int>()
        (0 until config.numberOfTargetAttributeClassesInDataSet).forEach {
            if (it == correctOutput) correctOutputs.add(1) else correctOutputs.add(0)
        }
        return correctOutputs.toList()
    }

    private fun activateInputLayer(inputs: List<Double>) {
        inputLayer.readInput(inputs)
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