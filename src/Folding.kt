class Folding (val dataSet : MutableList<List<Double>>, val k : Int) {

    val folds : MutableList<MutableList<List<Double>>> = mutableListOf()
    val numberOfSets : Int = dataSet.count()
    var lastIndex : Int = 0

    init {
        foldDataSet()
    }

    private fun foldDataSet() {
        var extra = getNumberOfExtraElementsInFold()
        for(i in 1..k) {
            folds.add(getIntervalsOfSets(i, extra))
            extra--
        }
    }

    private fun getIntervalsOfSets(i : Int, extra : Int) : MutableList<List<Double>> {
        return if(extra > 0) {
            val fromIndex = lastIndex
            val toIndex = fromIndex+Math.floor(getNumberOfElementsInFold()).toInt()+1
            lastIndex = toIndex
            dataSet.subList(fromIndex, toIndex)
        }
        else {
            val fromIndex = lastIndex
            val toIndex = fromIndex+Math.floor(getNumberOfElementsInFold()).toInt()
            lastIndex = toIndex
            dataSet.subList(fromIndex,toIndex)
        }
    }

    private fun getNumberOfElementsInFold() : Double {
        return numberOfSets/k.toDouble();
    }

    private fun getNumberOfExtraElementsInFold() : Int {
        return numberOfSets%k;
    }
}

fun main(args : Array<String>) {
    val dr = DataReader("./data/haberman.data")
    val cv = Folding(dr.getTrainingDataSet(), 10)
    println(cv.folds[0])
    println(cv.folds[1])
    println(cv.folds[2])
    println(cv.folds[3])
    println(cv.folds[4])
    println(cv.folds[5])
    println(cv.folds[6])
    println(cv.folds[7])
    println(cv.folds[8])
    println(cv.folds[9])
}