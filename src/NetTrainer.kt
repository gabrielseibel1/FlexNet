class NetTrainer (private val flexNet: FlexNet) {

    fun train(fold: Fold) {
        for (instance in fold.dataSet) {
            println(instance)
            for (i in 1..50) flexNet.propagateAndBackPropagate(instance)
            flexNet.print()
        }
    }
}