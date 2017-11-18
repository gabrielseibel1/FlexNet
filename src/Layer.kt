class Layer (size: Int, thetasPerNeuron: Int) {

    private val neurons : List<Neuron>
    val activations = mutableListOf<Double>()

    init {
        val initNeurons = mutableListOf<Neuron>()
        for (index in 1..size) {
            initNeurons.add(Neuron(thetasPerNeuron))
        }
        neurons = initNeurons.toList()
    }

    fun activate (previousLayer: Layer): List<Double> {
        activations.clear()
        neurons.forEach { activations.add( it.activate(previousLayer) ) }
        return activations
    }

    fun readInput(inputs: List<Double>) {
        if (inputs.size != neurons.size)
            throw Exception("Invalid inputs! Expected size ${neurons.size} but got ${inputs.size}")
        activations.clear()
        activations.addAll(inputs)
    }

    override fun toString(): String = buildString {
        append("Layer{ size = ${neurons.size}, ")

        if (activations.isEmpty())
            append("No activations ")
        else
            append("Activations ${activations.joinToString(prefix = "{", postfix = "}")}")


        append(", ")
        neurons.forEach { append(it) ; append(" ") }
        append("}")
    }
}