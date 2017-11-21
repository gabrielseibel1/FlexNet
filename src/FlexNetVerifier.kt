class FlexNetVerifier {

    private val config = FlexNetConfig(
            inputNeurons = 1,
            hiddenLayers = 2,
            neuronsPerHiddenLayer = 2,
            numberOfTargetAttributeClassesInDataSet = 1,
            lambda = 1.0,
            alpha = 0.1
    )

    fun verify(instances: List<Instance>) {
        val epsilon = 0.01
        val flexNet = FlexNet(config)

        instances.forEach { flexNet.propagate(it.attributes) ; flexNet.backPropagate(it.targetAttributeNeuron) }
        //println(config) ; println("\n")
        flexNet.print() ; println("\n")

        flexNet.getAllThetas().forEachIndexed { layerIndex, layer -> //iterate over each layer's neurons' theta-list
            if (layerIndex != 0) //don't check gradient on input layer
                layer.forEachIndexed { neuronIndex, neuron -> //iterate over each neuron's theta-list
                    neuron.forEachIndexed { thetaIndex, theta -> //iterate over each theta of a neuron's theta-list

                        val backPropagationGradOfJWithRespectToTheta = flexNet.getGrad(layerIndex, neuronIndex, thetaIndex)
                        //println("J(θ) = ${flexNet.calculateJ(instances)}")

                        //apply (theta+ε) to the network and numerically calculate it's J
                        flexNet.changeOneTheta(layerIndex, neuronIndex, thetaIndex, theta + epsilon)
                        val jOfThetaPlusEpsilon = flexNet.calculateJ(instances)
                        //println("J(θ+ε) = $jOfThetaPlusEpsilon")

                        //apply (theta-ε) to the network and numerically calculate it's J
                        flexNet.changeOneTheta(layerIndex, neuronIndex, thetaIndex, theta - epsilon)
                        val jOfThetaMinusEpsilon = flexNet.calculateJ(instances)
                        //println("J(θ-ε) = $jOfThetaMinusEpsilon")

                        //numerically calculate gradient of J with respect to theta
                        val numericGradOfJWithRespectToTheta = (jOfThetaPlusEpsilon - jOfThetaMinusEpsilon) / (2*epsilon)

                        //restore theta to it's original value and check grad from back propagation
                        flexNet.changeOneTheta(layerIndex, neuronIndex, thetaIndex, theta)


                        println("∂J(θ)/∂θ for theta[$layerIndex][$neuronIndex][$thetaIndex]: $numericGradOfJWithRespectToTheta")
                        println("Back-propagation gradient for theta[$layerIndex][$neuronIndex][$thetaIndex]: $backPropagationGradOfJWithRespectToTheta \n")
                    }
                }
        }
    }

}

fun main(args: Array<String>) {
    val verifier = FlexNetVerifier()
    val instance1 = Instance(mutableListOf(0.1), "Class1", 0)
    val instance2 = Instance(mutableListOf(0.2), "Class2", 1)
    val instance3 = Instance(mutableListOf(0.3), "Class1", 0)
    val instance4 = Instance(mutableListOf(0.4), "Class2", 1)
    val instance5 = Instance(mutableListOf(0.5), "Class1", 0)
    val instance6 = Instance(mutableListOf(0.6), "Class2", 1)
    val instances = mutableListOf(instance1/*, instance2, instance3, instance4, instance5, instance6*/)
    verifier.verify(instances)
}