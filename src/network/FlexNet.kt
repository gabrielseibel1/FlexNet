package network

class FlexNet (private val config : FlexNetConfig) {

    private val inputLayer : Layer = Layer(config.inputNeurons, 0, isInputLayer = true)
    private val outputLayer : Layer = Layer(config.numberOfTargetAttributeClassesInDataSet, config.neuronsPerHiddenLayer, isOutputLayer = true)
    private val hiddenLayers : List<Layer>
    private var numberOfNetThetas : Int = 0
    private var JFunction : Double = 0.0


    init {
        val initLayers = mutableListOf<Layer>()
        for (index in 1..config.hiddenLayers) {
            val neuronsThetaCount = if (index == 1) config.inputNeurons else config.neuronsPerHiddenLayer
            initLayers.add(Layer(config.neuronsPerHiddenLayer, neuronsThetaCount))
        }
        hiddenLayers= initLayers.toList()
        calculateNumberOfNetThetas()
    }

    fun changeOneTheta(layerIndex: Int, neuronIndex: Int, thetaIndex: Int, value: Double) {
        val layer: Layer
        val layerDescription:String
        when {
            layerIndex == 0 -> throw Exception("You don't want to change a theta from the input layer!")

            (layerIndex > 0) and (layerIndex-1 <= hiddenLayers.lastIndex) -> {
                layer = hiddenLayers[layerIndex-1]
                layerDescription = "hiddenLayer ${layerIndex-1}"
            }

            else -> {
                layer = outputLayer
                layerDescription = "outputLayer"
            }
        }

        if ((!layer.isOutputLayer) and (neuronIndex == layer.neurons.lastIndex)) throw Exception("Bias neuron doesn't have a gradient!")

        val neuron = layer.neurons[neuronIndex]
        //println("Changing theta[$layerDescription][neuron $neuronIndex][theta $thetaIndex] from ${neuron.thetas[thetaIndex]} to $value")
        neuron.thetas[thetaIndex] = value
    }

    fun getGrad(layerIndex: Int, neuronIndex: Int, thetaIndex: Int): Double {
        val layer: Layer
        val previousLayer: Layer
        val layerDescription:String
        when {
            layerIndex == 0 -> throw Exception("Input layer's neurons don't have a gradient!")

            (layerIndex > 0) and (layerIndex-1 <= hiddenLayers.lastIndex) -> {
                layer = hiddenLayers[layerIndex-1]
                previousLayer = if (layerIndex == 1) /*layer is first of the hidden*/ inputLayer else hiddenLayers[layerIndex-2]
                layerDescription = "hiddenLayer ${layerIndex-1}"
            }

            else -> {
                layer = outputLayer
                previousLayer = hiddenLayers.last()
                layerDescription = "outputLayer"
            }
        }

        if ((!layer.isOutputLayer) and (neuronIndex == layer.neurons.lastIndex)) throw Exception("Bias neuron doesn't have a gradient!")

        val neuron = layer.neurons[neuronIndex]
        val theta = neuron.thetas[thetaIndex]

        //don't use regularization term for bias-neuron's theta
        val grad: Double
        val gradDescription: String
        if (thetaIndex == previousLayer.neurons.lastIndex) {
            grad =  previousLayer.neurons[thetaIndex].activation * neuron.delta
            gradDescription = "${previousLayer.neurons[thetaIndex].activation} * ${neuron.delta}"
        }
        else {
            grad = previousLayer.neurons[thetaIndex].activation*neuron.delta + config.lambda*theta
            gradDescription = "${previousLayer.neurons[thetaIndex].activation} * ${neuron.delta} + ${config.lambda}*$theta"
        }

        println("Theta[$layerDescription][neuron $neuronIndex][theta $thetaIndex] = $theta")
        //println("Gradient = $gradDescription = $grad")

        return grad
    }

    private fun getAllActivations(): List<List<Double>> {
        val allActivations = mutableListOf<List<Double>>()
        hiddenLayers.forEach { allActivations.add(it.getActivations()) }
        allActivations.add(outputLayer.getActivations())
        return allActivations.toList()
    }

    fun getAllThetas(): List<List<List<Double>>> {
        val allThetas = mutableListOf<List<List<Double>>>()
        allThetas.add(inputLayer.getThetasOfEachNeuron())
        hiddenLayers.forEach { allThetas.add(it.getThetasOfEachNeuron()) }
        allThetas.add(outputLayer.getThetasOfEachNeuron())
        return allThetas.toList()
    }

    private fun getOutputs(): List<Double> {
        val outputs = mutableListOf<Double>()
        outputLayer.neurons.forEach {
                outputs.add(it.activation)
        }
        return outputs.toList()
    }

    fun getPredictedClass() : Int {
        var predicted = 0
        var probability = 0.0
        outputLayer.neurons.forEachIndexed { index, output ->
                if (output.activation > probability) {
                    probability = output.activation
                    predicted = index
                }
        }
        return predicted
    }

    fun getNumberOfClasses() : Int = outputLayer.neurons.count()

    private fun calculateNumberOfNetThetas() {
        for (neuron in inputLayer.neurons) {
            numberOfNetThetas+=neuron.thetas.count()
        }
        for (neuron in outputLayer.neurons) {
            numberOfNetThetas+=neuron.thetas.count()
        }
        hiddenLayers
                .flatMap { it.neurons }
                .forEach { numberOfNetThetas+= it.thetas.count() }
    }

    fun calculateJ(folding: Folding, foldTest : Int): Double {
        //add instances from test fold to a list of instances
        val testingInstances = mutableListOf<Instance>()
        folding.folds[foldTest].dataSet.forEach { testingInstances.add(it) }

        return calculateJ(testingInstances)
    }

    fun calculateJ(instances: List<Instance>): Double {
        JFunction = 0.0
        var count = 0
        var regularization = 0.0

        //calculate first term
        instances.forEach {
            count++
            propagate(it.attributes)
            val correctOutputs = buildCorrectOutputs(it.targetAttributeNeuron)
            val outputs = getOutputs()
            //println("Outputs $outputs")
            //println("Correct outputs $correctOutputs")
            for (k in 0..correctOutputs.lastIndex) {
                JFunction += -correctOutputs[k]*(Math.log(outputs[k]))  -  (1-correctOutputs[k])*Math.log(1-outputs[k])
            }
        }
        JFunction /= count

        //calculate regularization term
        var previousLayer = inputLayer
        hiddenLayers.forEach { layer ->
            layer.neurons.forEach {
                it.thetas.forEachIndexed { index, theta ->
                    if (index != previousLayer.neurons.lastIndex) regularization += Math.pow(theta, 2.0)
                }
            }
            previousLayer = layer
        }
        outputLayer.neurons.forEach {
            it.thetas.forEachIndexed { index, theta ->
                if (index != previousLayer.neurons.lastIndex) regularization += Math.pow(theta, 2.0)
            }
        }
        regularization = (regularization*config.lambda)/(2*count)

        //join two terms
        JFunction += regularization
        return JFunction
    }

    fun forthAndBackPropagate(instance: Instance) {
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

    fun backPropagate(correctNeuronToActivate: Int) {
        val correctOutputs = buildCorrectOutputs(correctNeuronToActivate)
        var previousLayer = outputLayer
        outputLayer.calculateDeltasFromCorrectOutputs(correctOutputs)
        hiddenLayers.asReversed().forEach {
            it.calculateDeltas(previousLayer)
            previousLayer = it
        }
    }

    fun updateThetas() {
        var previousLayer = inputLayer
        hiddenLayers.forEach {
            it.updateThetas(previousLayer, config.alpha, config.lambda)
            previousLayer = it
        }
        outputLayer.updateThetas(previousLayer, config.alpha, config.lambda)
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