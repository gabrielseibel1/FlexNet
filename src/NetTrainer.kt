class NetTrainer (private val flexNet: FlexNet) {

    fun train(preFold: MutableList<List<Double>>, classIsFirstAttribute: Boolean) {
        val fold = Fold(preFold, convert = true)
        for (instance in fold.instances) {
            println(instance)
            if (classIsFirstAttribute)
                flexNet.propagate(instance.attributes.subList(1, instance.attributes.count()))
            else
                flexNet.propagate(instance.attributes.subList(0, instance.attributes.count()-1))
            flexNet.print()
        }
    }
}