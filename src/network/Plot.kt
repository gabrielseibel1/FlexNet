package network

import org.math.plot.*
import javax.swing.JFrame

class Plot(x: DoubleArray, y: DoubleArray, private val title: String = "A network.Plot") {

    private val plot = Plot2DPanel()

    constructor(points: MutableList<Pair<Double, Double>>, title: String = "A network.Plot"):
            this(x = points.getXDoubleArray(), y = points.getYDoubleArray(), title = title)

    init {
        // add a line plot to the PlotPanel
        plot.addLinePlot("my plot", x, y)
    }

    fun show() {
        // put the PlotPanel in a JFrame, as a JPanel
        val frame = JFrame(title)
        frame.contentPane = plot
        frame.isVisible = true
    }
}

/**
 * Return DoubleArray from "first" entries of pairs
 */
private fun MutableList<Pair<Double, Double>>.getXDoubleArray(): DoubleArray {
    val xList = mutableListOf<Double>()
    this.forEach { xList.add(it.first) }
    return xList.toDoubleArray()
}

/**
 * Return DoubleArray from "second" entries of pairs
 */
private fun MutableList<Pair<Double, Double>>.getYDoubleArray(): DoubleArray {
    val yList = mutableListOf<Double>()
    this.forEach { yList.add(it.second) }
    return yList.toDoubleArray()
}

fun main(args: Array<String>) {

    val x = doubleArrayOf(1.0, 2.0, 3.0)
    val y = doubleArrayOf(1.0, 4.0, 9.0)

    val plot = Plot(x, y)
    plot.show()
}