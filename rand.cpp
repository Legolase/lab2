#include "rand.h"
#include <random>

template<typename T>
static constexpr T sqr(T a) {
    return a * a;
}

template<typename T>
static constexpr T power(T a, size_t n) {
    return n == 0 ? 1 : sqr(power(a, n / 2)) * (n % 2 == 0 ?  1 : a);
}

int random(long long begin, long long end)
{
    //static std::mt19937 gen { std::random_device{}() };
    static std::mt19937 gen{3};
    static std::uniform_int_distribution<long long> dist(0, std::numeric_limits<long long>::max());
    return (dist(gen) % (end - begin + 1)) + begin;
}

double random(double begin, double end, unsigned int precision)
{
    const int divisor = power(10, (precision > 20) ? 20 : precision);
    long long i_begin = static_cast<long long>(begin * divisor);
    long long i_end = static_cast<long long>(end * divisor);
    return random(i_begin, i_end) / static_cast<double>(divisor);
}
