class Layer (size: Int, thetasPerNeuron: Int) {

    private val neurons : List<Neuron>
    val activations = mutableListOf<Double>()

    constructor(inputs: List<Double>) : this(inputs.size, 0) {
        activations.addAll(inputs)
    }

    init {
        val init_neurons = mutableListOf<Neuron>()
        for (index in 1..size) {
            init_neurons.add(Neuron(thetasPerNeuron))
        }
        neurons = init_neurons.toList()
    }

    fun activate (previousLayer: Layer): List<Double> {
        activations.clear()
        neurons.forEach { activations.add( it.activate(previousLayer) ) }
        return activations
    }

    override fun toString(): String = buildString {
        append("Layer{ size = ${neurons.size}, ")

        if (activations.isEmpty())
            append("No activations ")
        else
            append("Activations ${activations.joinToString()}")


        append(", ")
        neurons.forEach { append(it) ; append(" ") }
        append("}")
    }
}