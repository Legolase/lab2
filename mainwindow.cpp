#include "mainwindow.h"
#include <cassert>
#include <set>
#include <tuple>
#include "rand.h"
#include <fstream>
#include <chrono>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    plot.setInteractions(QCP::iRangeDrag | QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    plot.axisRect()->setupFullAxesBox(true);
    plot.xAxis->setLabel("k");
    plot.yAxis->setLabel("b");

    start();

    plot.legend->setVisible(true);
    resize(1400, 1000);
    setCentralWidget(&plot);
}

void MainWindow::start()
{
    QString func_str("y = %1x + %2");
    QString leg_str("Result (batch %1)");
    qreal k, b, lrk, lrb, dlt;
    int mx_step, n;
    std::ifstream fin(".\\file.input");

    if (!fin) {
        qDebug() << "file doesn't exists";
    }
    fin >> k >> b >> lrk >> lrb >> dlt >> mx_step >> n;
    fin.close();
    qDebug() << "In:" << k << b << lrk << lrb << dlt << mx_step;

    auto points = get_points_by_line(500, k, b, 0, 10, 2);
    auto started = std::chrono::high_resolution_clock::now();
//    set_points(points, "Points");
//    set_line(k, b, "Expected");

//    auto result = linear_regression(points, 1, lrk, lrb, k, b, mx_step, dlt);
//    qDebug() << std::get<2>(result) << func_str.arg(std::get<0>(result)).arg(std::get<1>(result));
    auto momentum_result = momentum_linear_regression(points, lrk, lrb, k, b, mx_step, dlt);
    auto nesterov_result = nesterov_linear_regression(points, lrk, lrb, k, b, mx_step, dlt);
    auto adagrad_result = adagrad_linear_regression(points, lrk, lrb, k, b, mx_step, dlt);
    auto rmsprop_result = rmsprop_linear_regression(points, lrk, lrb, k, b, mx_step, dlt);
    auto adam_result = adam_linear_regression(points, lrk, lrb, k, b, mx_step, dlt);
//    qDebug() << result.size() - 1 << func_str.arg(result.back().key).arg(result.back().value);

    auto done = std::chrono::high_resolution_clock::now();
    qDebug() << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();

//    set_line(result.back().key, result.back().value, "Result");
//    set_line(std::get<0>(result), std::get<1>(result), "Result");

    auto f = [&points] (qreal const k, qreal const b) {
        return mse(points, k, b);
    };

    set_color_map({-7.5, -2.5}, {7.5, 12.5}, {500, 500}, f);
    make_way(momentum_result, "Momentum");
    make_way(nesterov_result, "Nesterov");
    make_way(adagrad_result, "AdaGrad");
    make_way(rmsprop_result, "RMSProp");
    make_way(adam_result, "Adam");
}

void MainWindow::set_color_map(QPointF const& left_bottom, QPointF const& right_top,
                               QSize const& resolution,
                               auto const& f)
{
    QCPColorMap* color_map = new QCPColorMap(plot.xAxis, plot.yAxis);
    plot.legend->removeItem(0);
    color_map->data()->setSize(resolution.width(), resolution.height());
    color_map->data()->setRange(QCPRange(left_bottom.x(), right_top.x()),
                                QCPRange(left_bottom.y(), right_top.y()));
    color_map->setTightBoundary(true);
    qreal dx, dy;

    // set heights
    QPoint cur{0, 0};
    for (cur.ry() = 0; cur.y() < resolution.height(); ++cur.ry()) {
        for (cur.rx() = 0; cur.x() < resolution.width(); ++cur.rx()) {
            color_map->data()->cellToCoord(cur.x(), cur.y(), &dx, &dy);
            color_map->data()->setCell(cur.x(), cur.y(), f(dx, dy));
        }
    }

    //set gradient
    QCPColorScale* color_scale = new QCPColorScale(&plot);
    color_map->setInterpolate(true);
    plot.plotLayout()->addElement(0, 1, color_scale);
    color_scale->setType(QCPAxis::atRight);
    color_scale->axis()->setLabel("MSE");



    color_map->setColorScale(color_scale);
    color_map->rescaleDataRange();

    QCPColorGradient grad(QCPColorGradient::gpCold);
    grad.setLevelCount(17);
    color_map->setGradient(grad);


    QCPMarginGroup *marginGroup = new QCPMarginGroup(&plot);
    plot.axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    color_scale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    color_map->rescaleAxes();
}

static QPen random_pen() {
    static int arr[] = {3, 7, 8, 9, 10, 11, 12, 5};
    static int cur = 1;
    cur %= 8;
    QPen pen = QPen(QBrush(static_cast<Qt::GlobalColor>(arr[cur++])), 1.5);
    return pen;
//    uchar main_colour = static_cast<uchar>(random(175, 255));
//    uchar sc1 = static_cast<uchar>(random(25, 150));
//    uchar sc2 = static_cast<uchar>(random(25, 150));
//    switch (random(0, 2)) {
//    case 0:
//        pen = QPen(QBrush(QColor(main_colour, sc1, sc2)), 1.5);
//        break;
//    case 1:
//        pen = QPen(QBrush(QColor(sc1, main_colour, sc2)), 1.5);
//        break;
//    case 2:
//        pen = QPen(QBrush(QColor(sc1, sc2, main_colour)), 1.5);
//        break;
//    }
//    return pen;
}

void MainWindow::make_way(v<QCPCurveData> const& way, QString const& name)
{
    QCPCurve *curve = new QCPCurve(plot.xAxis, plot.yAxis);
    curve->setPen(random_pen());
    curve->setName(name);
    curve->data()->set(way, true);
//    plot.addGraph();
//    plot.graph()->addData(way[0].key, way[0].value);
//    plot.graph()->setName("Start");
//    plot.graph()->setPen(QPen(QBrush(),0));

//    QCPScatterStyle sc_begin(QCPScatterStyle::ssCross);
//    sc_begin.setPen(QPen(QBrush(Qt::green),2));
//    plot.graph()->setScatterStyle(sc_begin);

//    plot.addGraph();
//    plot.graph()->addData(way.back().key, way.back().value);
//    plot.graph()->setName("End");
//    plot.graph()->setPen(QPen(QBrush(),0));

//    QCPScatterStyle sc_end(QCPScatterStyle::ssTriangle);
//    sc_end.setPen(QPen(QBrush(Qt::red),2));
//    plot.graph()->setScatterStyle(sc_end);
}

void MainWindow::set_points(const way_t &points, QString const& name)
{
    plot.addGraph();
    plot.graph()->setScatterStyle(QCPScatterStyle::ssDisc);
    plot.graph()->setPen(Qt::NoPen);
    plot.graph()->setName(name);
    for (auto const& elem : points) {
        plot.graph()->addData(elem.first, elem.second);
    }
    plot.rescaleAxes();
    auto x_range = plot.xAxis->range(), y_range = plot.yAxis->range();
    qreal x_diff = x_range.upper - x_range.lower, y_diff = y_range.upper - y_range.lower;
    plot.xAxis->setRange(x_range.lower - 0.05 * x_diff, x_range.upper + 0.05 * x_diff);
    plot.yAxis->setRange(y_range.lower - 0.05 * y_diff, y_range.upper + 0.05 * y_diff);
}

void MainWindow::set_line(const qreal k, const qreal b, QString const& name)
{
    auto f = [&k, &b](qreal const x) { return k * x + b; };
    QPen pen = random_pen();
    qreal const x_min = plot.xAxis->range().lower;
    qreal const x_max = plot.xAxis->range().upper;
    plot.addGraph();
    plot.graph()->setPen(pen);
    plot.graph()->addData({x_min, x_max}, {f(x_min), f(x_max)});
    plot.graph()->setName(name);
}


way_t get_points_by_line(int n, const qreal k, const qreal b, const qreal x_min, const qreal x_max, const qreal delta)
{
    way_t points;
    qreal x_temp;
    while (n-- > 0) {
        x_temp = random(x_min, x_max, 5);
        points.emplace_back(x_temp, k * x_temp + b + random(-delta, delta, 5));
    }
    return points;
}
