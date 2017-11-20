class FlexNet (private val config : FlexNetConfig) {

    private val inputLayer : Layer = Layer(config.inputNeurons, 1)
    private val outputLayer : Layer = Layer(config.numberOfTargetAttributeClassesInDataSet, config.neuronsPerHiddenLayer)
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

    fun calculateJ(folds: MutableList<Fold>, foldTest : Int) : Double {
        var count = 0
        var regularization = 0.0
        folds.forEachIndexed {
            index, fold -> run {
                if(index != foldTest-1) {
                    fold.dataSet.forEach {
                        instance -> run {
                            count++
                            propagate(instance.attributes)
                            val correctOutputs = buildCorrectOutputs(instance.targetAttributeNeuron)
                            for (output in 1..correctOutputs.count()) {
                                JFunction += -correctOutputs[output-1]*(Math.log(outputLayer.neurons[output-1].activation))-(1-correctOutputs[output-1])*Math.log(1-outputLayer.neurons[output-1].activation)
                            }
                        }
                    }
                }
            }
        }
        JFunction = JFunction/count
        inputLayer.neurons.forEach {
            neuron -> run {
                neuron.thetas.forEachIndexed {
                    index, theta ->
                        if(index != inputLayer.neurons.lastIndex) regularization += Math.pow(theta, 2.0)
                }
            }
        }
        hiddenLayers.forEach {
            layer -> run {
                layer.neurons.forEach {
                    neuron -> run {
                        neuron.thetas.forEachIndexed {
                            index, theta -> if(index != layer.neurons.lastIndex) regularization += Math.pow(theta, 2.0)
                        }
                    }
                }
            }
        }
        outputLayer.neurons.forEach {
            neuron -> run {
                neuron.thetas.forEachIndexed {
                    index, theta -> if(index != outputLayer.neurons.lastIndex) regularization += Math.pow(theta, 2.0)
                }
            }
        }
        regularization = (regularization*config.lambda)/(2*count)
        JFunction += regularization
        //println(count)
        //println("JFunction = $JFunction")
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

    private fun backPropagate(correctOutput: Int) {
        val correctOutputs = buildCorrectOutputs(correctOutput)
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