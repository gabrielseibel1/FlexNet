package network

data class Metrics(
            val j : Double,
            val accuracy : Double,
            val precision : Double,
            val recall : Double,
            val standardDeviationJ : Double,
            val standardDeviationAccuracy : Double,
            val standardDeviationPrecision : Double,
            val standardDeviationRecall : Double
)