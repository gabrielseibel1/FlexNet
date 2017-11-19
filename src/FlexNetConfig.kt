data class FlexNetConfig (
        val hiddenLayers: Int,
        val neuronsPerHiddenLayer: Int,
        val inputNeurons: Int,
        val numberOfTargetAttributeClassesInDataSet: Int,
        /**
         * Regularization factor
         */
        val lambda: Double = 0.5,
        /**
         * Step of the thetas updating factor
         */
        val alpha: Double = 0.5
)