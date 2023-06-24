#include "algos.h"
#include "rand.h"
#include <QDebug>
#include <cmath>

#define DEBUG_OUTPUT 0

qreal mse(v<pr<qreal, qreal>> const& points, qreal const k, qreal const b) noexcept {
    auto f = [&k, &b](qreal const x) {
        return k * x + b;
    };

    qreal result = 0, diff;

    for (auto const& elem : points) {
        diff = elem.second - f(elem.first);
        result += diff * diff;
    }
    return result / points.size();
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

std::tuple<qreal, qreal, int> linear_regression(const way_t &points, const int batch, const qreal lrk, const qreal lrb, const qreal k, const qreal b, const int max_step, const qreal dlt)
{
    assert(batch > 0 && batch <= points.size());
    qreal optimal_sse = mse(points, k, b);
    pr<qreal, qreal> cur = {0, 0};
    qreal cur_mse;

    for (int i = 1; i <= max_step; ++i) {
        cur = step(points, cur.first, cur.second, lrk, lrb, batch);
        if (std::isnan(cur.first) || std::isnan(cur.second)) {
            return {cur.first, cur.second, i};
        }
#if DEBUG_OUTPUT
        if (i % 10 == 0) {
            qDebug() << cur.first << cur.second;
        }
#endif
        cur_mse = mse(points, cur.first, cur.second);

        if (std::abs(optimal_sse - cur_mse) < dlt) {
            return {cur.first, cur.second, i};
        }
    }

    return {cur.first, cur.second, max_step};
}

pr<qreal, qreal> step(const way_t &points, qreal const k, qreal const b, qreal const ck, qreal const cb, int const batch) {
    auto chosen_index = rand_seq(batch, points.size());

    qreal gradk = 0;
    qreal gradb = 0;
    qreal temp;

    for (auto const& elem : chosen_index) {
        temp = points[elem].second - (points[elem].first * k) - b;
        gradk += (-2. / points.size()) * temp * points[elem].first;
        gradb += (-2. / points.size()) * temp;
    }

    return {k - ck * gradk, b - cb * gradb};
}

std::tuple<qreal, qreal, int> sdg_linear_regression(
    const way_t &points,
    const qreal lrk,
    const qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt)
{
    qreal mse_bs = mse(points, k, b);
    qreal cur_mse = 0;
    qreal const coef = 2. / (points.size() + 1);
    pr<qreal, qreal> cur = {0, 0};
    qreal diff;
    auto f = [&cur](qreal const x) {
        return cur.first * x + cur.second;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    for (int i = 0; i < max_step; ++i) {
        diff = y(i) - f(x(i));
        cur.first -= lrk  * (-2.) * diff * x(i);
        cur.second -= lrb  * (-2.) * diff;
        diff *= diff;

        mse_bs = coef * diff + (1 - coef) * mse_bs;
        cur_mse = coef * diff + (1 - coef) * cur_mse;

#if DEBUG_OUTPUT
        if (i % 100 == 0) {
            qDebug() << mse_bs << cur_mse << std::abs(mse_bs - cur_mse);
        }
#endif

        if (std::abs(mse_bs - cur_mse) < dlt) {
            return {cur.first, cur.second, i};
        }
    }
    return {cur.first, cur.second, max_step};
}

v<QCPCurveData> momentum_linear_regression(
    const way_t &points,
    const qreal lrk,
    const qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt)
{
    qreal const m_force = 0.5;
    qreal const optimal_mse = mse(points, k, b);

    qreal b_force = 0;
    qreal cur_mse;
    qreal diff;
    qreal k_force = 0;
    qreal gradk;
    qreal gradb;

    pr<qreal, qreal> cur = {0, 0};

    auto f = [&cur](qreal const x) {
        return cur.first * x + cur.second;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    v<QCPCurveData> way = {{0, cur.first, cur.second}};

    for (int i = 1; i <= max_step; ++i) {
        diff = y(i) - f(x(i));
        gradk = -2. * diff * x(i);
        gradb = -2. * diff;
        k_force = m_force * k_force - lrk * gradk;
        b_force = m_force * b_force - lrb * gradb;
        cur.first += k_force;
        cur.second += b_force;

        cur_mse = mse(points, cur.first, cur.second);

        way.emplace_back(i, cur.first, cur.second);
        if (std::abs(cur_mse - optimal_mse) < dlt) {
            break;
        }
    }
    return way;
}

v<QCPCurveData> nesterov_linear_regression(const way_t &points, const qreal lrk, const qreal lrb, const qreal k, const qreal b, const int max_step, const qreal dlt)
{
    qreal const m_force = 0.6;
    qreal const optimal_mse = mse(points, k, b);

    qreal cur_mse;
    qreal b_force = 0;
    qreal k_force = 0;
    qreal gradk;
    qreal gradb;

    pr<qreal, qreal> cur = {0, 0};
    auto f = [&cur, &lrk, &k_force, &lrb, &b_force](qreal const x) {
        return (cur.first - lrk * k_force) * x + cur.second - lrb * b_force;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    v<QCPCurveData> way = {{0, cur.first, cur.second}};

    for (int i = 0; i < max_step; ++i) {
        gradk = -2. * (y(i) - f(x(i))) * x(i);
        gradb = -2. * (y(i) - f(x(i)));
        k_force = m_force * k_force - lrk * gradk;
        b_force = m_force * b_force - lrb * gradb;
        cur.first += k_force;
        cur.second += b_force;

        cur_mse = mse(points, cur.first, cur.second);
        way.emplace_back(i, cur.first, cur.second);
        if (std::abs(cur_mse - optimal_mse) < dlt) {
            break;
        }
    }
    return way;
}

v<QCPCurveData> adagrad_linear_regression(
    const way_t &points,
    qreal lrk,
    qreal lrb,
    const qreal k,
    const qreal b,
    const int max_step,
    const qreal dlt)
{
    qreal const optimal_mse = mse(points, k, b);
    lrk *= 250;
    lrb *= 150;
    qreal gk = 0;
    qreal gb = 0;
    qreal gradk;
    qreal gradb;
    qreal diff;
    qreal cur_mse;

    pr<qreal, qreal> cur = {0, 0};
    auto f = [&cur](qreal const x) {
        return cur.first * x + cur.second;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    v<QCPCurveData> way = {{-1, cur.first, cur.second}};

    for (int i = 0; i < max_step; ++i) {
        diff = y(i) - f(x(i));
        gradk = -2. * diff * x(i);
        gradb = -2. * diff;
        gk += gradk * gradk;
        gb += gradb * gradb;
        cur.first -= (lrk / std::sqrt(gk + dlt)) * gradk;
        cur.second -= (lrb / std::sqrt(gb + dlt)) * gradb;

        cur_mse = mse(points, cur.first, cur.second);

        way.emplace_back(i, cur.first, cur.second);
        if (std::abs(cur_mse - optimal_mse) < dlt) {
            break;
        }
    }

    return way;
}

v<QCPCurveData> rmsprop_linear_regression(const way_t &points, qreal const lrk, qreal const lrb, const qreal k, const qreal b, const int max_step, const qreal dlt)
{
    qreal const optimal_mse = mse(points, k, b);
    qreal const pwr = 0.9;
    qreal gk = 0;
    qreal gb = 0;
    qreal diff;
    qreal gradk;
    qreal gradb;
    qreal cur_mse;

    pr<qreal, qreal> cur = {0, 0};
    auto f = [&cur](qreal const x) {
        return cur.first * x + cur.second;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    v<QCPCurveData> way = {{-1, cur.first, cur.second}};

    for (int i = 0; i < max_step; ++i) {
        diff = y(i) - f(x(i));
        gradk = -2. * diff * x(i);
        gradb = -2. * diff;
        gk = pwr * gk + (1 - pwr) * gradk * gradk;
        gb = pwr * gb + (1 - pwr) * gradb * gradb;
        cur.first -= (lrk / std::sqrt(gk + dlt))*gradk;
        cur.second -= (lrb / std::sqrt(gb + dlt))*gradb;
        cur_mse = mse(points, cur.first, cur.second);

        way.emplace_back(i, cur.first, cur.second);
        if (std::abs(cur_mse - optimal_mse) < dlt) {
            break;
        }
    }
    return way;
}

v<QCPCurveData> adam_linear_regression(const way_t &points, qreal const lrk, qreal const lrb, const qreal k, const qreal b, const int max_step, const qreal dlt)
{
    qreal const optimal_mse = mse(points, k, b);
    qreal const pwr1 = 0.9;
    qreal const pwr2 = 0.98;
    qreal p1k = 0;
    qreal p1b = 0;
    qreal p2k = 0;
    qreal p2b = 0;
    qreal diff;
    qreal gradk;
    qreal gradb;
    qreal cur_mse;

    pr<qreal, qreal> cur = {0, 0};
    auto f = [&cur](qreal const x) {
        return cur.first * x + cur.second;
    };
    auto x = [&points](int const i) { return points[i % points.size()].first;};
    auto y = [&points](int const i) { return points[i % points.size()].second;};

    v<QCPCurveData> way = {{0, cur.first, cur.second}};

    for (int i = 1; i < max_step; ++i) {
        diff = y(i) - f(x(i));
        gradk = -2. * diff * x(i);
        gradb = -2. * diff;
        p1k = pwr1 * p1k + (1 - pwr1) * gradk;
        p1b = pwr1 * p1b + (1 - pwr1) * gradb;
        p2k = pwr2 * p2k + (1 - pwr2) * gradk * gradk;
        p2b = pwr2 * p2b + (1 - pwr2) * gradb * gradb;
        p1k /= 1. - std::pow(pwr1, i);
        p1b /= 1. - std::pow(pwr1, i);
        p2k /= 1. - std::pow(pwr2, i);
        p2b /= 1. - std::pow(pwr2, i);
        cur.first -= (lrk / std::sqrt(p2k + dlt))*gradk;
        cur.second -= (lrb / std::sqrt(p2b + dlt))*gradb;
        cur_mse = mse(points, cur.first, cur.second);

        way.emplace_back(i, cur.first, cur.second);
        if (std::abs(cur_mse - optimal_mse) < dlt) {
            break;
        }
    }
    return way;
}

v<pr<qreal, qreal> > get_points_polynomial(int n, auto const& f, const qreal x_min, const qreal x_max, const qreal delta)
{
    v<pr<qreal, qreal>> points(n);

    for (auto& elem : points) {
        elem.first = random(x_min, x_max, 5);
        elem.second = f(elem.first + random(-delta, delta, 5));
    }

    return points;
}


qreal poly_mse(const v<pr<qreal, qreal> > &points, const v<qreal> &params) noexcept
{
    auto f = [&params](qreal const x) {
        qreal result = 0;

        for (int i = 0; i < params.size(); ++i) {
            result += params[i] * std::pow(x, i);
        }
        return result;
    };

    qreal result = 0, diff;

    for (auto const& elem : points) {
        diff = elem.second - f(elem.first);
        result += diff * diff;
    }
    return result / points.size();
}

v<qreal> polynomial_regression(const way_t &points, const int degree, auto const& regulation)
{

}
