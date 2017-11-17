import java.util.*

class Neuron (thetaCount: Int){

    private val thetas : MutableList<Double> = mutableListOf()

    init {
        //initialize thetas with 0
        for (index in 1..thetaCount) {
            thetas.add(0.0)
        }
    }

    fun activate(previousLayer : Layer) : Double {
        var sum = 0.0
        previousLayer.activations.forEachIndexed {
            index, activation -> sum += thetas[index] * activation
        }
        return sum
    }

    override fun toString(): String = buildString {
        append("Neuron{ thetas: ")
        append(thetas.joinToString())
        append(" }")
    }
}