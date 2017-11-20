class NetTrainer {

    var confusionMatrix : MutableList<MutableList<Int>> = mutableListOf()

    fun train(flexNet: FlexNet, fold: Fold) {
        for (instance in fold.dataSet) {
            flexNet.forthAndBackPropagate(instance)
            flexNet.updateThetas()
        }
    }

    fun calculateConfusionMatrix(flexNet: FlexNet, fold: Fold) {
        confusionMatrix = mutableListOf()
        for(i in 1..flexNet.getNumberOfClasses()) {
            var auxList : MutableList<Int> = mutableListOf()
            for(j in 1..flexNet.getNumberOfClasses()) {
                auxList.add(0)
            }
            confusionMatrix.add(auxList)
        }
        for (instance in fold.dataSet) {
            println(instance)
            flexNet.propagate(instance.attributes)
            println(flexNet.getPredictedClass())
            confusionMatrix[instance.targetAttributeNeuron][flexNet.getPredictedClass()]++
        }
        println(confusionMatrix)
    }

    fun getAccuracy(flexNet: FlexNet) : Double {
        var accuracy = 0.0
        var total = 0
        for(i in 1..flexNet.getNumberOfClasses()) {
            for (j in 1..flexNet.getNumberOfClasses()) {
                total += confusionMatrix[i-1][j-1]
                if(i == j) accuracy += confusionMatrix[i-1][j-1]
            }
        }
        return accuracy/total
    }

    fun getPrecision(flexNet: FlexNet) : Double {
        var precisao = 0.0
        for(i in 1..flexNet.getNumberOfClasses()) {
            var vp = 0
            var vn = 0
            var fp = 0
            var fn = 0
            for(trueClass in 1..flexNet.getNumberOfClasses()) {
                for (predictedClass in 1..flexNet.getNumberOfClasses()) {
                    if(trueClass == i && trueClass == predictedClass) vp += confusionMatrix[trueClass-1][predictedClass-1]
                    if(trueClass == i && trueClass != predictedClass) fn += confusionMatrix[trueClass-1][predictedClass-1]
                    if(predictedClass == i && trueClass != predictedClass) fp += confusionMatrix[trueClass-1][predictedClass-1]
                    if(trueClass != i && trueClass != predictedClass) vn += confusionMatrix[trueClass-1][predictedClass-1]
                }
            }
            if(fp+vp == 0) {
                precisao +=0
            }
            else {
                precisao += vp/(fp+vp).toDouble()
            }
        }
        return precisao/flexNet.getNumberOfClasses()
    }

    fun getRecall(flexNet: FlexNet) : Double {
        var recall = 0.0
        for(i in 1..flexNet.getNumberOfClasses()) {
            var vp = 0
            var vn = 0
            var fp = 0
            var fn = 0
            for(trueClass in 1..flexNet.getNumberOfClasses()) {
                for (predictedClass in 1..flexNet.getNumberOfClasses()) {
                    if(trueClass == i && trueClass == predictedClass) vp += confusionMatrix[trueClass-1][predictedClass-1]
                    if(trueClass == i && trueClass != predictedClass) fn += confusionMatrix[trueClass-1][predictedClass-1]
                    if(predictedClass == i && trueClass != predictedClass) fp += confusionMatrix[trueClass-1][predictedClass-1]
                    if(trueClass != i && trueClass != predictedClass) vn += confusionMatrix[trueClass-1][predictedClass-1]
                }
            }
            if(fn+vp == 0) recall +=0
            else recall += vp/(fn+vp).toDouble()
        }
        return recall/flexNet.getNumberOfClasses()
    }
}

fun main(args: Array<String>) {
    val config = FlexNetConfig(
            inputNeurons = 1,
            numberOfTargetAttributeClassesInDataSet = 2,
            hiddenLayers = 1,
            neuronsPerHiddenLayer = 3
    )
    val flexNet = FlexNet(config)
    val netTrainer = NetTrainer()
    flexNet.print()
    val instance1 = Instance(mutableListOf(0.1/*,0.9*/), "Class1", 0)
    val instance2 = Instance(mutableListOf(0.2/*,0.8*/), "Class2", 1)
    val instances = listOf(instance1, instance2)
    netTrainer.train(flexNet, Fold(instances))
}