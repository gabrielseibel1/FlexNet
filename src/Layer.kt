class Layer (size: Int, thetasPerNeuron: Int) {

    val neurons: List<Neuron>

    init {
        val initNeurons = mutableListOf<Neuron>()

        for (index in 1..size) {
            initNeurons.add(Neuron(thetasPerNeuron + 1 /*sum one to account for bias theta (theta0) */))
        }
        val biasNeuron = Neuron(0)
        biasNeuron.activation = 1.0
        initNeurons.add(biasNeuron)

        neurons = initNeurons.toList()
    }

    fun readInput(inputs: List<Double>) {
        val expectedSize = neurons.size - 1 //neurons have an extra entry for bias
        if (inputs.size != expectedSize)
            throw Exception("Invalid inputs! Expected size $expectedSize but got ${inputs.size}")
        neurons
                .filterIndexed{ index, _ ->  index != neurons.lastIndex } //don't change bias neuron
                .forEachIndexed{ index, neuron -> neuron.activation = inputs[index] }
    }

    fun activate(previousLayer: Layer) {
        neurons.forEachIndexed { index, neuron ->
            //don't activate bias neuron
            if (index != neurons.lastIndex)
                neuron.activate(previousLayer)
        }
    }

    fun calculateDeltasFromCorrectOutputs(correctOutputs: List<Int>) {
        neurons.forEachIndexed{ index, neuron ->
            //don't calculate delta for bias neuron
            if (index != neurons.lastIndex)
               neuron.calculateDelta(correctOutputs[index])
        }
    }

    fun calculateDeltas(nextLayer: Layer) {
        neurons.forEachIndexed{ index, neuron ->
            //don't calculate delta for bias neuron
            if (index != neurons.lastIndex)
                neuron.calculateDelta(nextLayer, index)
        }
    }

    fun updateThetas(previousLayer: Layer, alpha: Double, lambda: Double) {
        neurons.forEachIndexed{ index, neuron ->
            //don't calculate theta for bias neuron (it has no thetas)
            if (index != neurons.lastIndex)
                neuron.updateThetas(previousLayer, alpha, lambda)
        }
    }

    override fun toString(): String = buildString {
        append("Layer(size: ${neurons.size}, neurons: ${neurons.joinToString(prefix = "{", postfix = "}")} )")
    }
}