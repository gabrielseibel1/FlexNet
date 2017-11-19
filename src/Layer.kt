class Layer (size: Int, thetasPerNeuron: Int) {

    val neurons : List<Neuron>

    init {
        val initNeurons = mutableListOf<Neuron>()
        for (index in 1..size) {
            initNeurons.add(Neuron(thetasPerNeuron))
        }
        neurons = initNeurons.toList()
    }

    fun readInput(inputs: List<Double>) {
        if (inputs.size != neurons.size)
            throw Exception("Invalid inputs! Expected size ${neurons.size} but got ${inputs.size}")
        neurons.forEachIndexed{ index, neuron -> neuron.activation = inputs[index] }
    }

    fun activate(previousLayer: Layer) {
        neurons.forEach { it.activate(previousLayer) }
    }

    fun calculateDeltasFromCorrectOutputs(correctOutputs: List<Int>) {
        neurons.forEachIndexed { index, neuron ->  neuron.calculateDelta(correctOutputs[index]) }
    }

    fun calculateDeltas(nextLayer: Layer) {
        neurons.forEachIndexed{ index, neuron ->  neuron.calculateDelta(nextLayer, index) }
    }

    fun updateThetas(previousLayer: Layer, alpha: Double, lambda: Double) {
        neurons.forEach{ it.updateThetas(previousLayer, alpha, lambda) }
    }

    override fun toString(): String = buildString {
        append("Layer(size: ${neurons.size}, neurons: ${neurons.joinToString(prefix = "{", postfix = "}")} )")
    }
}