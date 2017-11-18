data class Fold (val instances: MutableList<Instance>) {
    constructor(preFold: MutableList<List<Double>>, convert: Boolean) : this(preFold.toInstances())
}

fun MutableList<List<Double>>.toInstances() : MutableList<Instance> {
    val instances: MutableList<Instance> = mutableListOf()
    this.forEach {
        instances.add(Instance(it))
    }
    return instances
}