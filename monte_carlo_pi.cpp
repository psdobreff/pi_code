// Estimate pi with a Monte Carlo method.
//
// Random points (x, y) are drawn uniformly from the unit square [0,1)^2.
// The fraction that falls inside the quarter circle x^2 + y^2 <= 1
// approaches pi/4, so pi ~= 4 * hits / samples.
//
// Each sample is a Bernoulli trial with p = pi/4, so the standard error
// of the estimate is 4 * sqrt(p * (1 - p) / n).
//
// Random numbers come from SplitMix64, a counter-based pseudo-random
// generator: the k-th value of the stream is mix(seed + k * gamma). Any
// chunk of samples can therefore jump straight to its place in the
// stream, so each batch is split into fixed-size chunks shared among
// OpenMP threads and the result for a given seed is identical no matter
// how many threads run or how the chunks are scheduled. The generator
// matches the Fortran version, so both give the same output for the
// same seed.
//
// Build:  g++ -std=c++17 -O2 -fopenmp -o monte_carlo_pi_cpp monte_carlo_pi.cpp
// Run:    OMP_NUM_THREADS=8 ./monte_carlo_pi_cpp [total_samples [seed]]
//
// If no seed is given, one is chosen at random and printed so the run
// can be repeated.

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <string_view>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::int64_t default_samples = 100'000'000;
constexpr std::int64_t batch_size = 1'000'000;
// Samples per unit of parallel work.
constexpr std::int64_t chunk_size = 4096;
constexpr double pi_ref = 3.141592653589793238462643383279502884;

struct Estimate {
    double pi;
    double std_err;
};

Estimate estimate(std::int64_t samples, std::int64_t inside) {
    const double frac = static_cast<double>(inside) / static_cast<double>(samples);
    return {4.0 * frac,
            4.0 * std::sqrt(frac * (1.0 - frac) / static_cast<double>(samples))};
}

// Return command-line argument number pos as an integer, or the fallback
// if it is absent. An invalid argument is an error: exit rather than
// silently running a different (possibly much longer) job.
template <typename T>
T read_int_arg(int argc, char* argv[], int pos, std::string_view what,
               T min_value, T fallback) {
    if (argc <= pos) return fallback;

    const std::string_view arg{argv[pos]};
    T value{};
    const auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc{} || ptr != arg.data() + arg.size() || value < min_value) {
        std::fprintf(stderr, "Error: invalid %.*s '%.*s' (expected an integer from %s to %s)\n",
                     static_cast<int>(what.size()), what.data(),
                     static_cast<int>(arg.size()), arg.data(),
                     std::to_string(min_value).c_str(),
                     std::to_string(std::numeric_limits<T>::max()).c_str());
        std::exit(EXIT_FAILURE);
    }
    return value;
}

// A non-repeatable 64-bit seed for when the user does not supply one.
std::uint64_t random_seed_value() {
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

// SplitMix64 constants. Unsigned arithmetic wraps modulo 2^64, which the
// generator relies on.
constexpr std::uint64_t sm_gamma = 0x9E3779B97F4A7C15;
constexpr std::uint64_t sm_mul1 = 0xBF58476D1CE4E5B9;
constexpr std::uint64_t sm_mul2 = 0x94D049BB133111EB;

// SplitMix64 output function: scramble a stream state into 64 random
// bits and keep the top 53 as a double in [0, 1).
double splitmix_unit(std::uint64_t z) {
    z = (z ^ (z >> 30)) * sm_mul1;
    z = (z ^ (z >> 27)) * sm_mul2;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// Draw m points starting at global sample index start and return how
// many fall inside the quarter circle. Sample j uses stream values
// 2j+1 and 2j+2, so the result depends only on the seed and the sample
// indices, never on which thread does the work.
std::int64_t count_hits(std::uint64_t seed, std::int64_t start, std::int64_t m) {
    std::uint64_t state = seed + 2 * static_cast<std::uint64_t>(start) * sm_gamma;
    std::int64_t inside = 0;
    for (std::int64_t k = 0; k < m; ++k) {
        state += sm_gamma;
        const double x = splitmix_unit(state);
        state += sm_gamma;
        const double y = splitmix_unit(state);
        if (x * x + y * y <= 1.0) ++inside;
    }
    return inside;
}

void report(std::int64_t samples, std::int64_t inside) {
    const auto [est, se] = estimate(samples, inside);
    std::printf("%15" PRId64 "  %14.10f  %12.4e  %12.4e\n",
                samples, est, std::abs(est - pi_ref), se);
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto total_samples = read_int_arg<std::int64_t>(
        argc, argv, 1, "sample count", 1, default_samples);
    // Only consult std::random_device when no seed was supplied.
    const auto seed = argc > 2
        ? read_int_arg<std::uint64_t>(argc, argv, 2, "seed", 0, 0)
        : random_seed_value();

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    std::printf("Estimating pi with %" PRId64 " random samples on %d thread(s)\n",
                total_samples, num_threads);
    std::printf("Seed: %" PRIu64 "\n", seed);
    std::printf("\n");
    std::printf("%15s  %14s  %12s  %12s\n",
                "samples", "pi estimate", "abs error", "std error");

    std::int64_t done = 0;
    std::int64_t hits = 0;
    std::int64_t next_report = 1000;

    while (done < total_samples) {
        // Stop each batch at the next report point so every power of 10
        // gets its own row, even below batch_size.
        const std::int64_t n =
            std::min({batch_size, total_samples - done, next_report - done});
#pragma omp parallel for schedule(static) reduction(+ : hits)
        for (std::int64_t i = 0; i < n; i += chunk_size) {
            hits += count_hits(seed, done + i, std::min(chunk_size, n - i));
        }
        done += n;

        // Report at each power of 10 (and at the end) to show convergence.
        if (done == next_report || done == total_samples) {
            report(done, hits);
            if (done == next_report) next_report *= 10;
        }
    }

    const auto [pi_est, std_err] = estimate(total_samples, hits);

    std::printf("\n");
    std::printf("Final estimate : %16.12f\n", pi_est);
    std::printf("Reference pi   : %16.12f\n", pi_ref);
    std::printf("Absolute error : %16.6e\n", std::abs(pi_est - pi_ref));
    std::printf("95%% interval   : [%16.12f, %16.12f]\n",
                pi_est - 1.96 * std_err, pi_est + 1.96 * std_err);
}
