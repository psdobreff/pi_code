// Estimate pi with a Monte Carlo method.
//
// Random points (x, y) are drawn uniformly from the unit square [0,1)^2.
// The fraction that falls inside the quarter circle x^2 + y^2 <= 1
// approaches pi/4, so pi ~= 4 * hits / samples.
//
// Each sample is a Bernoulli trial with p = pi/4, so the standard error
// of the estimate is 4 * sqrt(p * (1 - p) / n).
//
// Build:  g++-16 -std=c++23 -O2 -o monte_carlo_pi_cpp monte_carlo_pi.cpp
// Run:    ./monte_carlo_pi [total_samples]

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <print>
#include <random>
#include <string_view>

namespace {

constexpr std::int64_t default_samples = 100'000'000;
constexpr double pi_ref = std::numbers::pi;

struct Estimate {
    double pi;
    double std_err;
};

Estimate estimate(std::int64_t samples, std::int64_t inside) {
    const double frac = static_cast<double>(inside) / static_cast<double>(samples);
    return {4.0 * frac,
            4.0 * std::sqrt(frac * (1.0 - frac) / static_cast<double>(samples))};
}

// Return the sample count from the first command-line argument, or the
// fallback if none is given. An invalid argument is an error: exit
// rather than silently running a different (possibly much longer) job.
std::int64_t read_sample_count(int argc, char* argv[], std::int64_t fallback) {
    if (argc < 2) return fallback;

    const std::string_view arg{argv[1]};
    std::int64_t value{};
    const auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc{} || ptr != arg.data() + arg.size() || value < 1) {
        std::println(stderr, "Error: invalid sample count '{}' "
                             "(expected a positive integer, e.g. 1000000)", arg);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

void report(std::int64_t samples, std::int64_t inside) {
    const auto [est, se] = estimate(samples, inside);
    std::println("{:15}  {:14.10f}  {:12.4e}  {:12.4e}",
                 samples, est, std::abs(est - pi_ref), se);
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::int64_t total_samples = read_sample_count(argc, argv, default_samples);

    // 64-bit Mersenne Twister pseudo-random number generator, seeded from
    // a non-deterministic source so each run differs.
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    std::println("Estimating pi with {} random samples", total_samples);
    std::println("");
    std::println("{:>15}  {:>14}  {:>12}  {:>12}",
                 "samples", "pi estimate", "abs error", "std error");

    std::int64_t hits = 0;
    std::int64_t next_report = 1000;

    for (std::int64_t done = 1; done <= total_samples; ++done) {
        const double x = uniform(rng);
        const double y = uniform(rng);
        if (x * x + y * y <= 1.0) ++hits;

        // Report at each power of 10 (and at the end) to show convergence.
        if (done == next_report || done == total_samples) {
            report(done, hits);
            next_report *= 10;
        }
    }

    const auto [pi_est, std_err] = estimate(total_samples, hits);

    std::println("");
    std::println("Final estimate : {:16.12f}", pi_est);
    std::println("Reference pi   : {:16.12f}", pi_ref);
    std::println("Absolute error : {:16.6e}", std::abs(pi_est - pi_ref));
    std::println("95% interval   : [{:16.12f}, {:16.12f}]",
                 pi_est - 1.96 * std_err, pi_est + 1.96 * std_err);
}
