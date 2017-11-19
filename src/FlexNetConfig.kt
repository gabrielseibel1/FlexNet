data class FlexNetConfig (
        val hiddenLayers: Int,
        val neuronsPerHiddenLayer: Int,
        val inputNeurons: Int,
        val numberOfTargetAttributeClassesInDataSet: Int,
        /**
         * Regularization factor
         */
        val lambda: Int
)