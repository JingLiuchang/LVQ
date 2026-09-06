#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ivf_lvq.h"

namespace {

thread_local std::string last_error;

class LvqP1Context {
  public:
    LvqP1Context(
        std::size_t dimension,
        std::uint32_t bits,
        const float *training_mean)
        : dimension_(dimension), bits_(bits) {
        if (dimension == 0 || training_mean == nullptr) {
            throw std::invalid_argument("dimension must be positive and training_mean must be non-null");
        }
        if (bits != 4 && bits != 8) {
            throw std::invalid_argument("LVQ C++ primary bits must be 4 or 8");
        }
        training_mean_.assign(training_mean, training_mean + dimension_);
    }

    double encode(
        const float *vectors,
        std::size_t vector_count,
        std::size_t dimension,
        std::size_t thread_count) {
        if (dimension != dimension_) {
            throw std::invalid_argument("encode dimension does not match the LVQ context");
        }
        if (vectors == nullptr && vector_count != 0) {
            throw std::invalid_argument("vectors must be non-null");
        }
        set_threads(thread_count);

        std::vector<float> zero_centroid(dimension_, 0.0f);
        std::vector<int> cluster_ids(vector_count, 0);
        const auto start = std::chrono::steady_clock::now();
        index_.build_from(
            vectors,
            zero_centroid.data(),
            cluster_ids.data(),
            vector_count,
            dimension_,
            1,
            bits_,
            0,
            training_mean_.data());
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(stop - start).count();
    }

    void distances(
        const float *queries,
        std::size_t query_count,
        std::size_t dimension,
        const std::int64_t *candidate_indices,
        std::size_t candidates_per_query,
        std::size_t thread_count,
        float *output) const {
        if (dimension != dimension_) {
            throw std::invalid_argument("query dimension does not match the LVQ context");
        }
        const std::size_t pair_count = query_count * candidates_per_query;
        if ((queries == nullptr || candidate_indices == nullptr || output == nullptr) && pair_count != 0) {
            throw std::invalid_argument("distance inputs must be non-null");
        }
        if (index_.N == 0 && pair_count != 0) {
            throw std::runtime_error("encode must be called before distances");
        }
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const std::int64_t index = candidate_indices[pair];
            if (index < 0 || static_cast<std::size_t>(index) >= index_.N) {
                throw std::out_of_range("candidate index is outside the encoded base set");
            }
        }
        set_threads(thread_count);

#pragma omp parallel for schedule(static)
        for (std::int64_t query_index = 0;
             query_index < static_cast<std::int64_t>(query_count);
             ++query_index) {
            std::vector<float> centered_query(dimension_);
            const float *query = queries + static_cast<std::size_t>(query_index) * dimension_;
            for (std::size_t dim = 0; dim < dimension_; ++dim) {
                centered_query[dim] = query[dim] - training_mean_[dim];
            }
            for (std::size_t candidate = 0; candidate < candidates_per_query; ++candidate) {
                const std::size_t pair = static_cast<std::size_t>(query_index) * candidates_per_query + candidate;
                const std::size_t data_index = static_cast<std::size_t>(candidate_indices[pair]);
                const std::uint8_t *record = index_.records.data() + data_index * index_.record_stride;
                output[pair] = lvq::lvq_distance_dispatch(
                    record,
                    centered_query.data(),
                    dimension_,
                    bits_,
                    0,
                    index_.primary_bytes,
                    false);
            }
        }
    }

    std::size_t code_bits() const {
        return dimension_ * bits_;
    }

    std::size_t logical_bytes() const {
        return 4 + lvq::packed_bytes(dimension_, bits_);
    }

    std::size_t memory_bytes() const {
        return index_.record_stride;
    }

  private:
    static void set_threads(std::size_t thread_count) {
#ifdef _OPENMP
        omp_set_num_threads(static_cast<int>(std::max<std::size_t>(1, thread_count)));
#else
        (void)thread_count;
#endif
    }

    std::size_t dimension_;
    std::uint32_t bits_;
    std::vector<float> training_mean_;
    lvq::IVFLVQ index_;
};

template <typename Function>
int guarded_call(Function &&function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return -1;
    }
}

} // namespace

extern "C" {

const char *lvq_p1_last_error() noexcept {
    return last_error.c_str();
}

void *lvq_p1_create(
    std::size_t dimension,
    std::uint32_t bits,
    const float *training_mean) noexcept {
    try {
        last_error.clear();
        return new LvqP1Context(dimension, bits, training_mean);
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void lvq_p1_destroy(void *context) noexcept {
    delete static_cast<LvqP1Context *>(context);
}

int lvq_p1_encode(
    void *context,
    const float *vectors,
    std::size_t vector_count,
    std::size_t dimension,
    std::size_t thread_count,
    double *elapsed_seconds) noexcept {
    return guarded_call([&] {
        if (context == nullptr || elapsed_seconds == nullptr) {
            throw std::invalid_argument("context and elapsed_seconds must be non-null");
        }
        *elapsed_seconds = static_cast<LvqP1Context *>(context)->encode(
            vectors, vector_count, dimension, thread_count);
    });
}

int lvq_p1_distances(
    void *context,
    const float *queries,
    std::size_t query_count,
    std::size_t dimension,
    const std::int64_t *candidate_indices,
    std::size_t candidates_per_query,
    std::size_t thread_count,
    float *output) noexcept {
    return guarded_call([&] {
        if (context == nullptr) {
            throw std::invalid_argument("context must be non-null");
        }
        static_cast<LvqP1Context *>(context)->distances(
            queries,
            query_count,
            dimension,
            candidate_indices,
            candidates_per_query,
            thread_count,
            output);
    });
}

std::size_t lvq_p1_code_bits(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<LvqP1Context *>(context)->code_bits();
}

std::size_t lvq_p1_logical_bytes(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<LvqP1Context *>(context)->logical_bytes();
}

std::size_t lvq_p1_memory_bytes(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<LvqP1Context *>(context)->memory_bytes();
}

} // extern "C"
