#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <utility>
#include <QVector>
#include "qcustomplot.h"

template<typename K, typename V>
using pr = std::pair<K, V>;

template<typename T>
using v = QVector<T>;

inline constexpr qreal EPSILON = 10.e-4;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

private:
    void start();

    void set_color_map(QPointF const& left_bottom, QPointF const& right_top,
                                   QSize const& resolution,
                       auto const& f);
    void make_way(v<QCPCurveData>const& way);
    void set_points(v<pr<qreal, qreal>> const& points, const char* name);
    void set_line(qreal const k, qreal const b, const char* name);

    QCustomPlot plot;
};

// random sequence generator of k elements from [0..n-1]
v<int> rand_seq(int const k, int const n);

//a function that returns how much the coefficients of the line k and b need to be changed {k_move, b_move}
pr<qreal, qreal> lr_step(v<pr<qreal, qreal>> const& points, qreal const k, qreal const b, qreal const lr, int const n);

qreal mse(v<pr<qreal, qreal>> const& points, qreal const k, qreal const b) noexcept;

v<QCPCurveData> linear_regression(v<pr<qreal, qreal>> const& points, pr<qreal, qreal> const& start, pr<qreal, qreal> const& expected, qreal const lr, int const n, int const max_step);

v<pr<qreal, qreal>> get_points_by_line(int n, qreal const k, qreal const b, qreal const x_min, qreal const x_max, qreal const delta);

#endif // MAINWINDOW_H
