#ifndef ALGOS_H
#define ALGOS_H

#include <QtGlobal>
#include <QVector>
#include "qcustomplot.h"

template<typename K, typename V>
using pr = std::pair<K, V>;

template<typename T>
using v = QVector<T>;

using way_t = v<pr<qreal, qreal>>;

// random sequence generator of k elements from [0..n-1]
v<int> rand_seq(int const k, int const n);

qreal mse(v<pr<qreal, qreal>> const& points, qreal const k, qreal const b) noexcept;

qreal poly_mse(v<pr<qreal, qreal>> const& points, v<qreal> const& params) noexcept;

// return {k, b, number_of_steps}
std::tuple<qreal, qreal, int> linear_regression(
    way_t const& points,
    int const batch,
    qreal const lrk,
    qreal const lrb,
    qreal const k,
    qreal const b,
    const int max_step,
    const qreal dlt
    );

//return new {k, b}
pr<qreal, qreal> step(
    const way_t &points,
    qreal const k,
    qreal const b,
    qreal const lrk,
    qreal const lrb,
    int const batch);

// return {k, b, number_of_steps}
std::tuple<qreal, qreal, int> sdg_linear_regression(
    way_t const& points,
    qreal const lrk,
    qreal const lrb,
    qreal const k,
    qreal const b,
    const int max_step,
    const qreal dlt
    );

v<QCPCurveData> momentum_linear_regression(
    const way_t &points,
    const qreal lrk,
    const qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt);

v<QCPCurveData> nesterov_linear_regression(
    const way_t &points,
    const qreal lrk,
    const qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt);

v<QCPCurveData> adagrad_linear_regression(
    const way_t &points,
    qreal lrk,
    qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt);

v<QCPCurveData> rmsprop_linear_regression(
    const way_t &points,
    qreal const lrk,
    qreal const lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt);

v<QCPCurveData> adam_linear_regression(
    const way_t &points,
    qreal const lrk,
    qreal const lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt);

v<pr<qreal, qreal>> get_points_polynomial(int n, auto const& f, qreal const x_min, qreal const x_max, qreal const delta);

v<qreal> polynomial_regression(
    const way_t &points,
    int const degree,
    auto const& regulation);

#endif // ALGOS_H
