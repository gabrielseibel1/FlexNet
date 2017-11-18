data class FlexNetConfig (
        val hiddenLayers: Int,
        val neuronsPerHiddenLayer: Int,
        val inputNeurons: Int,
        val outputNeurons: Int,
        /**
         * Regularization factor
         */
        val lambda: Int
)