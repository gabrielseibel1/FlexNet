package network

class NetTrainer(val stepOfJCheck: Int = 50,
                 private val MAX_TRIES:Int = 10000,
                 private val MAX_NO_IMPROVEMENT_TRIES: Int = 30,
                 private val MIN_J_DIFFERENCE_PERCENTAGE:Double = 0.01,
                 private val MIN_GOOD_J: Double = 0.000065) {

    var triesCounter: Int = 0
    var noImprovementCounter = 0
    var confusionMatrix : MutableList<MutableList<Int>> = mutableListOf()

    fun resetTriesCounter() {
        triesCounter = 0
    }

    /**
     * Trains a data set and returns if the training is done (true) or should train more (false)
     */
    fun trainAndCatalogJ(flexNet: FlexNet, trainingInstances: List<Instance>, testInstances: List<Instance>,
                         points: MutableList<Pair<Double, Double>>): Boolean {
        var previousJ: Double
        var newJ: Double

        //separate instances in (trainingInstances.size/stepOfJCheck) small batches
        (0 until (trainingInstances.size/stepOfJCheck)).forEach {
            //train batch
            previousJ = flexNet.calculateJ(testInstances)
            val batch = buildBatch(trainingInstances, it)
            trainBatch(flexNet, batch)
            newJ = flexNet.calculateJ(testInstances)
            //println("Count: $it | J: $newJ")

            //catalogs J and number of instances trained
            if (points.isEmpty()) {
                points.add(Pair(batch.size.toDouble(), newJ))
                //println(Pair(batch.size.toDouble(), newJ))
            } else {
                points.add(Pair(points.last().first + stepOfJCheck, newJ))
                //println(Pair(points.last().first + stepOfJCheck, newJ))
            }

            //after each batch is trained, checks if should end training
            if (shouldEndTraining(previousJ, newJ)) return true
        }

        //trains rest of instances that didn't fit in batches (if there is any)
        if (trainingInstances.size % stepOfJCheck != 0) {
            val remainingBatch = trainingInstances.subList(trainingInstances.size - (trainingInstances.size % stepOfJCheck), trainingInstances.size)
            if (remainingBatch.isNotEmpty()) {
                previousJ = flexNet.calculateJ(testInstances)
                trainBatch(flexNet, remainingBatch)
                newJ = flexNet.calculateJ(testInstances)

                //catalogs J and number of instances trained
                points.add(Pair(points.last().first + stepOfJCheck, newJ))
                //println(Pair(points.last().first + stepOfJCheck, newJ))

                //after rest of instances are trained, checks if should end training
                if (shouldEndTraining(previousJ, newJ)) return true
            }
        }

        //should train more, training is not done and training set was fully used
        return false
    }

    /**
     * Trains a data set and returns if the training is done (true) or should train more (false)
     */
    fun trainInstances(flexNet: FlexNet, trainingInstances: List<Instance>, testInstances: List<Instance>) : Boolean {
        var previousJ: Double
        var newJ: Double

        //separate instances in (trainingInstances.size/stepOfJCheck) small batches
        (0 until (trainingInstances.size/stepOfJCheck)).forEach {
            //train batch
            previousJ = flexNet.calculateJ(testInstances)
            val batch = buildBatch(trainingInstances, it)
            trainBatch(flexNet, batch)
            newJ = flexNet.calculateJ(testInstances)

            //after each batch is trained, checks if should end training
            if (shouldEndTraining(previousJ, newJ)) return true
        }

        //trains rest of instances that didn't fit in batches (if there is any)
        if (trainingInstances.size % stepOfJCheck != 0) {
            val remainingBatch = trainingInstances.subList(trainingInstances.size - (trainingInstances.size % stepOfJCheck), trainingInstances.size)
            if (remainingBatch.isNotEmpty()) {
                previousJ = flexNet.calculateJ(testInstances)
                trainBatch(flexNet, remainingBatch)
                newJ = flexNet.calculateJ(testInstances)

                //after rest of instances are trained, checks if should end training
                if (shouldEndTraining(previousJ, newJ)) return true
            }
        }

        //should train more, training is not done and training set was fully used
        return false
    }

    /**
     * Trains a data set and returns if the training is done (true) or should train more (false)
     */
    fun trainFolding(flexNet: FlexNet, folding: Folding, testFoldIndex: Int) : Boolean {

        //add instances from training folds to one big list of instances called trainingInstances
        val trainingFolds = folding.folds.filterIndexed{ index, _ -> index != testFoldIndex }
        val trainingInstances = mutableListOf<Instance>()
        trainingFolds.forEach { it.dataSet.forEach { trainingInstances.add(it) } }

        //add instances from testing fold to one big list of instances called testingInstances
        val testFold = folding.folds[testFoldIndex]
        val testingInstances = mutableListOf<Instance>()
        testFold.dataSet.forEach { testingInstances.add(it) }

        return trainInstances(flexNet, testInstances = testingInstances, trainingInstances = trainingInstances)
    }

    /**
     * Trains a batch of instances
     */
    private fun trainBatch(flexNet: FlexNet, batch: List<Instance>) {
        batch.forEach{
            flexNet.forthAndBackPropagate(it)
            flexNet.updateThetas()
            //println("Predicted class is ${flexNet.getPredictedClass()}")
        }
    }

    /**
     * Builds a sublist of size stepOfJCheck, given a batch number, which determines where to cut the list
     */
    private fun buildBatch(instances: List<Instance>, batchNumber: Int): List<Instance> =
            instances.subList(batchNumber*stepOfJCheck, (batchNumber+1)*stepOfJCheck)

    private fun shouldEndTraining(previousJ: Double, newJ: Double): Boolean =
            isJGoodEnough(newJ) or triedEnoughTimes() or isStuckNotImproving(previousJ, newJ)

    private fun isStuckNotImproving(previousJ: Double, newJ: Double): Boolean {

        if (Math.abs((previousJ - newJ)/previousJ) <= MIN_J_DIFFERENCE_PERCENTAGE)
            noImprovementCounter++
        else
            noImprovementCounter = 0

        return if (noImprovementCounter >= MAX_NO_IMPROVEMENT_TRIES) {
            //println("Training stopped: low J difference percentage (${Math.abs((previousJ - newJ)/previousJ)})")
            true
        } else false
    }

    private fun isJGoodEnough(newJ: Double): Boolean {
        return if (newJ <= MIN_GOOD_J) {
            //println("Training stopped: found good J ($newJ)")
            true
        } else false
    }

    private fun triedEnoughTimes(): Boolean {
        triesCounter++
        if (triesCounter >= MAX_TRIES) {
            //println("Training stopped: took too many tries ($triesCounter)")
            return true
        } else
            return false

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
            //println(instance)
            flexNet.propagate(instance.attributes)
            //println("Target Attribute: "+instance)
            //println("Classe predita: "+flexNet.getPredictedClass()+" Classe esperada: "+instance.targetAttributeNeuron)
            //println((flexNet.getPredictedClass()==instance.targetAttributeNeuron))
            confusionMatrix[instance.targetAttributeNeuron][flexNet.getPredictedClass()]++
        }
        //println(confusionMatrix)
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
            if (fp+vp == 0) {
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
            neuronsPerHiddenLayer = 3,
            alpha = 0.0000001,
            lambda = 0.0
    )
    val flexNet = FlexNet(config)
    val netTrainer = NetTrainer()
    flexNet.print()
    val instance1 = Instance(mutableListOf(0.1), "Class1", 0)
    val instance2 = Instance(mutableListOf(0.2), "Class2", 1)
    val instance3 = Instance(mutableListOf(0.3), "Class1", 0)
    val instance4 = Instance(mutableListOf(0.4), "Class2", 1)
    val instance5 = Instance(mutableListOf(0.5), "Class1", 0)
    val instance6 = Instance(mutableListOf(0.6), "Class2", 1)

    netTrainer.trainFolding(flexNet, Folding(mutableListOf(instance1, instance2, instance3, instance4, instance5, instance6), 5), 1)
}