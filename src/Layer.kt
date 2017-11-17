class Layer (size: Int) {

    private val neurons : List<Neuron>
    val activations = mutableListOf<Double>()

    init {
        val init_neurons = mutableListOf<Neuron>()
        for (index in 1..size) {
            init_neurons.add(Neuron())
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
            append("No activations, ")
        else
            append("Activations ${activations.joinToString()}")

        neurons.forEach { append(it) }
        append("}")
    }
}