import org.math.plot.*
import javax.swing.JFrame

class Plot(x: DoubleArray, y: DoubleArray) {

    private val plot = Plot2DPanel()

    init {
        // add a line plot to the PlotPanel
        plot.addLinePlot("my plot", x, y)
    }

    fun show() {
        // put the PlotPanel in a JFrame, as a JPanel
        val frame = JFrame("a plot panel")
        frame.contentPane = plot
        frame.isVisible = true
    }
}

fun main(args: Array<String>) {

    val x = doubleArrayOf(1.0, 2.0, 3.0)
    val y = doubleArrayOf(1.0, 4.0, 9.0)

    val plot = Plot(x, y)
    plot.show()
}