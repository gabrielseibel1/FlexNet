class NetTrainer (private val flexNet: FlexNet) {

    fun train(preFold: MutableList<List<Double>>) {
        val fold = Fold(preFold, convert = true)
        for (instance in fold.instances) {
            println(instance)
            flexNet.propagate(instance.attributes.subList(0, instance.attributes.count()-1))
            flexNet.print()
        }
    }
}