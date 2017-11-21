import org.math.plot.*
import javax.swing.JFrame

fun main(args: Array<String>) {

    val x = doubleArrayOf(1.0, 2.0, 3.0)
    val y = doubleArrayOf(1.0, 2.0, 3.0)

    // create your PlotPanel (you can use it as a JPanel)
    val plot = Plot2DPanel()

    // add a line plot to the PlotPanel
    plot.addLinePlot("my plot", x, y)

    // put the PlotPanel in a JFrame, as a JPanel
    val frame = JFrame("a plot panel")
    frame.contentPane = plot
    frame.isVisible = true
}