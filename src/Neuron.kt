import java.util.*

class Neuron (thetaCount: Int){

    private val thetas : MutableList<Double> = mutableListOf()

    init {
        for (index in 1..thetaCount) {
            thetas.add((1..20).random().toDouble()/10)
        }
    }

    fun activate(previousLayer : Layer) : Double {
        var sum = 0.0
        previousLayer.activations.forEachIndexed {
            index, activation -> sum += thetas[index] * activation
        }
        return sum
    }

    private fun sigmoid(x: Double) : Nothing = TODO("Implement sigmoid function to use in activate()")

    override fun toString(): String = buildString {
        append("Neuron{ thetas: ${thetas.joinToString(prefix = "{", postfix = "}")}")
    }
}

fun ClosedRange<Int>.random() = start + Random().nextInt(endInclusive - start)