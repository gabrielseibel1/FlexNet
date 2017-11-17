class Neuron {

    private val thetas : MutableList<Double> = mutableListOf()

    init {
        //TODO init thetas aleatoriamente

    }

    fun activate(previousLayer : Layer) : Double {
        var sum = 0.0
        previousLayer.activations.forEachIndexed {
            index, activation -> sum += thetas[index] * activation
        }
        return sum
    }

    override fun toString(): String = buildString {
        append("Neuron()")
    }
}

