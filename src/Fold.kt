data class Fold (val instances: List<Instance>) {
    constructor(preFold: MutableList<List<Double>>, classIsFirstAttribute: Boolean) : this(preFold.toInstances(classIsFirstAttribute))
}

fun MutableList<List<Double>>.toInstances(classIsFirstAttribute: Boolean) : List<Instance> {
    val instances: MutableList<Instance> = mutableListOf()
    this.forEach {
        val targetAttribute: Int =  if (classIsFirstAttribute) it.first().toInt() else it.last().toInt()
        instances.add(Instance(it, targetAttribute))
    }
    return instances.toList()
}