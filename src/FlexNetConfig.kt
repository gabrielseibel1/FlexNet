data class FlexNetConfig (
        var hiddenLayers: Int,
        var neuronsPerHiddenLayer: Int,
        val inputNeurons: Int,
        val numberOfTargetAttributeClassesInDataSet: Int,
        /**
         * Regularization factor
         */
        var lambda: Double = 0.5,
        /**
         * Step of the thetas updating factor
         */
        var alpha: Double = 0.5
)