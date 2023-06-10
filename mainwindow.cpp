#include "mainwindow.h"
#include <cassert>
#include <set>
#include "rand.h"
#include <QDebug>

template<typename T>
using s = std::set<T>;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    plot.setInteractions(QCP::iRangeDrag | QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    plot.axisRect()->setupFullAxesBox(true);
    plot.xAxis->setLabel("k");
    plot.yAxis->setLabel("b");

    start();

    plot.legend->setVisible(true);
    resize(700, 600);
    setCentralWidget(&plot);
}

void MainWindow::start()
{
//    v<pr<qreal, qreal>> points = {
//        {1, -1},
//        {2, 0}
//    };

//    v<QCPCurveData> way = linear_regression(points, 0.01, 1, 10000);
//    make_way(way);
    qreal const k{-0.5}, b{4};
    auto points = get_points_by_line(20, k, b, 0, 10, 0.25);
//    set_points(points, "Points");
//    set_line(k, b, "Expected line");

    auto f = [&points](qreal k, qreal b) noexcept -> qreal {
        return mse(points, k, b)*100;
    };

    set_color_map({-3, -1}, {3, 5}, {500, 500}, f);
    auto way = linear_regression(points, {-2, 0.5}, {k, b}, 0.01, 20, 10000);
    make_way(way);
//    for (int i = 0; i < way.size()-1; ++i) {
//        if ((i < 100 && i % 10 == 0) || i % 50 == 0) {
//            set_line(way[i].key, way[i].value, "proc line");
//        }
//    }
    //set_line(way.back().key, way.back().value, "result");
}


v<int> rand_seq(int const k, int const n)
{
    assert(k <= n && k >= 0);
    v<int> elem(n), result;
    int temp;
    result.reserve(k);

    for (int i = 0; i < n; ++i) {
        elem[i] = i;
    }

    for (int i = 0; i < k; ++i) {
        temp = random(0, static_cast<int>(elem.size()) - 1);
        result.push_back(elem[temp]);
        elem[temp] = elem.back();
        elem.pop_back();
    }

    return result;
}

pr<qreal, qreal> lr_step(
    const v<pr<qreal, qreal> > &points, qreal const k,
    qreal const b,
    qreal const lr,
    int const n)
{
    assert(n > 0 && n <= points.size());
    v<int> chosen_set = rand_seq(n, points.size());
    qreal gradk{0}, gradb{0}, temp;

    for (int i = 0; i < n; ++i) {
        temp = points[chosen_set[i]].second - (k * points[chosen_set[i]].first + b);
        gradk += points[chosen_set[i]].first * temp;
        gradb += temp;
    }
    gradk *= (-2.) / n;
    gradb *= (-2.) / n;

    return { k - gradk * lr, b - gradb * lr };
}

qreal mse(const v<pr<qreal, qreal> > &points, const qreal k, const qreal b) noexcept
{
    auto f = [&k, &b](qreal x) noexcept {
        return k * x + b;
    };

    qreal result{0}, temp;

    for (auto const& elem : points) {
        temp = elem.second - f(elem.first);
        result += temp * temp;
    }

    return result / points.size();
}

v<QCPCurveData> linear_regression(
    const v<pr<qreal, qreal> > &points,
    pr<qreal, qreal> const& start,
    pr<qreal, qreal> const& expected,
    qreal const lr,
    int const n,
    int const max_step)
{
    v<QCPCurveData> way = {QCPCurveData(0, start.first, start.second)};
    pr<qreal, qreal> cur = {way.back().key, way.back().value};
    qreal f_cur;
    qreal f_expected = mse(points, expected.first, expected.second);

    qDebug() << "Start:" << cur.first << cur.sec дond;
    for (int i = 1; i <= max_step; ++i) {
        cur = lr_step(points, cur.first, cur.second, lr, n);
        f_cur = mse(points, cur.first, cur.second);

        if (std::abs(f_cur - f_expected) <= EPSILON) {
            break;
        }
        qDebug() << "Added: " << i << cur.first << cur.second;
        way.emplace_back(i, cur.first, cur.second);
    }
    qDebug() << "==============================";
    qDebug() << "Number of steps:" << way.size() - 1;
    return way;
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

void MainWindow::make_way(v<QCPCurveData> const& way)
{
    QCPCurve *curve = new QCPCurve(plot.xAxis, plot.yAxis);
    curve->setPen(QPen(QBrush(Qt::blue),2));
    curve->setName("Way");
    curve->data()->set(way, true);
    plot.addGraph();
    plot.graph()->addData(way[0].key, way[0].value);
    plot.graph()->setName("Start");
    plot.graph()->setPen(QPen(QBrush(),0));

    QCPScatterStyle sc_begin(QCPScatterStyle::ssCross);
    sc_begin.setPen(QPen(QBrush(Qt::green),2));
    plot.graph()->setScatterStyle(sc_begin);

    plot.addGraph();
    plot.graph()->addData(way.back().key, way.back().value);
    plot.graph()->setName("End");
    plot.graph()->setPen(QPen(QBrush(),0));

    QCPScatterStyle sc_end(QCPScatterStyle::ssTriangle);
    sc_end.setPen(QPen(QBrush(Qt::red),2));
    plot.graph()->setScatterStyle(sc_end);
}

void MainWindow::set_points(const v<pr<qreal, qreal> > &points, const char* name)
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

void MainWindow::set_line(const qreal k, const qreal b, const char* name)
{
    auto f = [&k, &b](qreal const x) { return k * x + b; };
    uchar main_colour = static_cast<uchar>(random(175, 255));
    uchar sc1 = static_cast<uchar>(random(25, 150));
    uchar sc2 = static_cast<uchar>(random(25, 150));
    QPen pen;
    switch (random(0, 2)) {
    case 0:
        pen = QPen(QBrush(QColor(main_colour, sc1, sc2)), 2.5);
        break;
    case 1:
        pen = QPen(QBrush(QColor(sc1, main_colour, sc2)), 2.5);
        break;
    case 2:
        pen = QPen(QBrush(QColor(sc1, sc2, main_colour)), 2.5);
        break;
    }
    qreal const x_min = plot.xAxis->range().lower;
    qreal const x_max = plot.xAxis->range().upper;
    plot.addGraph();
    plot.graph()->setPen(pen);
    plot.graph()->addData({x_min, x_max}, {f(x_min), f(x_max)});
    plot.graph()->setName(name);
}

// 1 2 7 9
// 9 1 7 2


v<pr<qreal, qreal> > get_points_by_line(int n, const qreal k, const qreal b, const qreal x_min, const qreal x_max, const qreal delta)
{
    v<pr<qreal, qreal>> points;
    qreal x_temp;
    while (n-- > 0) {
        x_temp = random(x_min, x_max, 5);
        points.emplace_back(x_temp, k * x_temp + b + random(-delta, delta, 5));
    }
    return points;
}
