import java.util.*

class Neuron (thetaCount: Int){

    private val thetas : MutableList<Double> = mutableListOf()
    var activation: Double = 0.0
    var delta: Double = 0.0

    init {
        //thetas random initialization
        for (index in 1..thetaCount) {
            thetas.add((-20..20).random().toDouble()/10)
        }
    }

    fun activate(previousLayer : Layer) {
        var sum = 0.0
        previousLayer.neurons.forEachIndexed { index, neuron -> sum += thetas[index] * neuron.activation }
        activation = sigmoid(sum)
    }

    private fun sigmoid(x: Double) : Double = 1 / (1 + Math.exp(-x))

    fun calculateDelta(correctOutput: Int) {
        delta = activation - correctOutput
    }

    fun calculateDelta(nextLayer: Layer, positionInCurrentLayer: Int) {
        var sum = 0.0
        nextLayer.neurons.forEachIndexed { index, neuron ->
            if (index != nextLayer.neurons.lastIndex)
                sum += neuron.thetas[positionInCurrentLayer] * neuron.delta
        }
        delta = sum * activation * (1 - activation)
    }

    fun updateThetas(previousLayer: Layer, alpha: Double) {
        previousLayer.neurons.forEachIndexed { index, neuron ->
            thetas[index] = thetas[index] - alpha * (neuron.activation * delta)
        }
    }

    override fun toString(): String = buildString {
        append("Neuron{")
        append(" thetas: ${thetas.joinToString(prefix = "{", postfix = "}")},")
        append(" activation: $activation,")
        append(" delta: $delta }")
    }
}

fun ClosedRange<Int>.random() = start + Random().nextInt(endInclusive - start)