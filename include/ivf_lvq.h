// IVF + Locally-adaptive Vector Quantization (LVQ) for P2 ANN experiments.
//
// This implementation follows Aguerrebere et al. 2023 for the LVQ storage and
// distance kernel, but uses an IVF coarse quantizer instead of a graph. Base
// vectors are assigned by a standard Faiss IVF/Flat coarse quantizer in Python;
// this C++ library stores and scans only LVQ-compressed residuals.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"

namespace lvq {

struct SearchStats {
    uint64_t scanned = 0;
    uint64_t reranked = 0;

    void add(const SearchStats& other) {
        scanned += other.scanned;
        reranked += other.reranked;
    }
};

struct Candidate {
    float dist = 0.0f;
    uint32_t id = 0;
    uint64_t pos = 0;

    bool operator<(const Candidate& other) const {
        return dist < other.dist;
    }
};

class FixedCandidateHeap {
public:
    explicit FixedCandidateHeap(uint64_t k) : k_(k) {}

    void push(float dist, uint32_t id, uint64_t pos) {
        if (k_ == 0) return;
        Candidate c{dist, id, pos};
        if (heap_.size() < k_) {
            heap_.push(c);
            return;
        }
        if (dist >= heap_.top().dist) return;
        heap_.pop();
        heap_.push(c);
    }

    float threshold() const {
        return heap_.size() == k_ ? heap_.top().dist
                                  : std::numeric_limits<float>::max();
    }

    std::vector<Candidate> sorted() {
        std::vector<Candidate> out;
        out.reserve(heap_.size());
        while (!heap_.empty()) {
            out.push_back(heap_.top());
            heap_.pop();
        }
        std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        });
        return out;
    }

    ResultHeap to_result_heap() {
        ResultHeap ret;
        while (!heap_.empty()) {
            const Candidate c = heap_.top();
            heap_.pop();
            ret.emplace(c.dist, c.id);
        }
        return ret;
    }

private:
    uint64_t k_ = 0;
    std::priority_queue<Candidate> heap_;
};

inline uint64_t round_up(uint64_t x, uint64_t align) {
    return ((x + align - 1) / align) * align;
}

inline uint64_t packed_bytes(uint64_t d, uint32_t bits) {
    return (d * static_cast<uint64_t>(bits) + 7U) / 8U;
}

inline void advise_huge_pages(void* data, size_t bytes) {
#if defined(__linux__)
    if (data == nullptr || bytes == 0) return;
    const long page_size_l = sysconf(_SC_PAGESIZE);
    if (page_size_l <= 0) return;
    const uintptr_t page_size = static_cast<uintptr_t>(page_size_l);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    const uintptr_t aligned_begin = begin & ~(page_size - 1U);
    const uintptr_t end = begin + static_cast<uintptr_t>(bytes);
    const uintptr_t aligned_end = (end + page_size - 1U) & ~(page_size - 1U);
    (void)madvise(
        reinterpret_cast<void*>(aligned_begin),
        static_cast<size_t>(aligned_end - aligned_begin),
        MADV_HUGEPAGE);
#else
    (void)data;
    (void)bytes;
#endif
}

template <class T>
inline void advise_huge_pages(std::vector<T>& vec) {
    advise_huge_pages(vec.data(), vec.size() * sizeof(T));
}

inline uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000U;
        uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp);
        if (shifted & 0x1000U) shifted += 0x2000U;
        return static_cast<uint16_t>(sign | (shifted >> 13));
    }
    if (exp >= 31) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7c00U);
        return static_cast<uint16_t>(sign | 0x7c00U | (mant >> 13) | 1U);
    }
    if (mant & 0x1000U) {
        mant += 0x2000U;
        if (mant & 0x800000U) {
            mant = 0;
            ++exp;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | (mant >> 13U));
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16U;
    uint32_t exp = (h >> 10U) & 0x1fU;
    uint32_t mant = h & 0x03ffU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1U;
                --exp;
            }
            mant &= 0x03ffU;
            exp = exp + (127U - 15U);
            bits = sign | (exp << 23U) | (mant << 13U);
        }
    } else if (exp == 31U) {
        bits = sign | 0x7f800000U | (mant << 13U);
    } else {
        exp = exp + (127U - 15U);
        bits = sign | (exp << 23U) | (mant << 13U);
    }

    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void store_half_pair(uint8_t* rec, float lo, float hi) {
    uint16_t lo_h = float_to_half(lo);
    uint16_t hi_h = float_to_half(hi);
    std::memcpy(rec, &lo_h, sizeof(uint16_t));
    std::memcpy(rec + 2, &hi_h, sizeof(uint16_t));
}

inline void load_half_pair(const uint8_t* rec, float& lo, float& hi) {
    uint16_t lo_h;
    uint16_t hi_h;
    std::memcpy(&lo_h, rec, sizeof(uint16_t));
    std::memcpy(&hi_h, rec + 2, sizeof(uint16_t));
    lo = half_to_float(lo_h);
    hi = half_to_float(hi_h);
}

inline uint8_t get_packed_code(const uint8_t* codes, uint64_t j, uint32_t bits) {
    if (bits == 8) return codes[j];
    const uint8_t byte = codes[j >> 1U];
    return (j & 1U) ? static_cast<uint8_t>(byte >> 4U)
                    : static_cast<uint8_t>(byte & 0x0fU);
}

inline void set_packed_code(uint8_t* codes, uint64_t j, uint32_t bits, uint8_t value) {
    if (bits == 8) {
        codes[j] = value;
        return;
    }
    uint8_t& byte = codes[j >> 1U];
    if (j & 1U) {
        byte = static_cast<uint8_t>((byte & 0x0fU) | ((value & 0x0fU) << 4U));
    } else {
        byte = static_cast<uint8_t>((byte & 0xf0U) | (value & 0x0fU));
    }
}

inline float sqr_l2(const float* a, const float* b, uint64_t d) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(av, bv);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(diff, diff, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
        av = _mm256_loadu_ps(a + i + 8);
        bv = _mm256_loadu_ps(b + i + 8);
        diff = _mm256_sub_ps(av, bv);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(diff, diff, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
    }
    for (; i + 8 <= d; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(av, bv);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(diff, diff, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
    }
    alignas(32) float lane[8];
    _mm256_store_ps(lane, acc);
    float s = lane[0] + lane[1] + lane[2] + lane[3] + lane[4] + lane[5] + lane[6] + lane[7];
    for (; i < d; ++i) {
        float v = a[i] - b[i];
        s += v * v;
    }
    return s;
#else
    float s = 0.0f;
    for (uint64_t i = 0; i < d; ++i) {
        const float v = a[i] - b[i];
        s += v * v;
    }
    return s;
#endif
}

template <uint64_t D>
inline float sqr_l2_fixed(const float* a, const float* b) {
    return sqr_l2(a, b, D);
}

inline float sqr_l2_dispatch(const float* a, const float* b, uint64_t d) {
    switch (d) {
        case 25: return sqr_l2_fixed<25>(a, b);
        case 50: return sqr_l2_fixed<50>(a, b);
        case 96: return sqr_l2_fixed<96>(a, b);
        case 128: return sqr_l2_fixed<128>(a, b);
        case 200: return sqr_l2_fixed<200>(a, b);
        case 512: return sqr_l2_fixed<512>(a, b);
        case 768: return sqr_l2_fixed<768>(a, b);
        case 960: return sqr_l2_fixed<960>(a, b);
        case 1024: return sqr_l2_fixed<1024>(a, b);
        default: return sqr_l2(a, b, d);
    }
}

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
    alignas(32) float lane[8];
    _mm256_store_ps(lane, v);
    return lane[0] + lane[1] + lane[2] + lane[3] + lane[4] + lane[5] + lane[6] + lane[7];
}

inline __m256 load_u8x8_as_ps(const uint8_t* p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m256i i = _mm256_cvtepu8_epi32(b);
    return _mm256_cvtepi32_ps(i);
}

inline __m256 load_nibble8_as_ps(const uint8_t* p) {
    uint32_t w;
    std::memcpy(&w, p, sizeof(uint32_t));
    return _mm256_setr_ps(
        static_cast<float>((w >> 0U) & 0x0fU),
        static_cast<float>((w >> 4U) & 0x0fU),
        static_cast<float>((w >> 8U) & 0x0fU),
        static_cast<float>((w >> 12U) & 0x0fU),
        static_cast<float>((w >> 16U) & 0x0fU),
        static_cast<float>((w >> 20U) & 0x0fU),
        static_cast<float>((w >> 24U) & 0x0fU),
        static_cast<float>((w >> 28U) & 0x0fU));
}

inline __m256 load_codes8_as_ps(const uint8_t* codes, uint64_t j, uint32_t bits) {
    return bits == 8 ? load_u8x8_as_ps(codes + j)
                     : load_nibble8_as_ps(codes + (j >> 1U));
}
#endif

inline float lvq_distance(
    const uint8_t* rec,
    const float* qres,
    uint64_t d,
    uint32_t primary_bits,
    uint32_t residual_bits,
    uint64_t primary_bytes,
    bool include_residual
) {
    float lo = 0.0f;
    float hi = 0.0f;
    load_half_pair(rec, lo, hi);
    const uint8_t* primary = rec + 4;
    const uint8_t* residual = primary + primary_bytes;

    const uint32_t primary_levels = 1U << primary_bits;
    const float delta = (hi > lo) ? ((hi - lo) / static_cast<float>(primary_levels - 1U)) : 0.0f;
    const bool use_residual = include_residual && residual_bits > 0 && delta > 0.0f;
    const uint32_t residual_levels = residual_bits > 0 ? (1U << residual_bits) : 0U;
    const float residual_lo = -0.5f * delta;
    const float residual_delta = use_residual
        ? (delta / static_cast<float>(residual_levels - 1U))
        : 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256 lo_v = _mm256_set1_ps(lo);
    const __m256 delta_v = _mm256_set1_ps(delta);
    const __m256 rlo_v = _mm256_set1_ps(residual_lo);
    const __m256 rdelta_v = _mm256_set1_ps(residual_delta);
    uint64_t j = 0;
    for (; j + 8 <= d; j += 8) {
        __m256 code = load_codes8_as_ps(primary, j, primary_bits);
#if defined(__FMA__)
        __m256 recon = _mm256_fmadd_ps(code, delta_v, lo_v);
#else
        __m256 recon = _mm256_add_ps(lo_v, _mm256_mul_ps(code, delta_v));
#endif
        if (use_residual) {
            __m256 rcode = load_codes8_as_ps(residual, j, residual_bits);
#if defined(__FMA__)
            recon = _mm256_add_ps(recon, _mm256_fmadd_ps(rcode, rdelta_v, rlo_v));
#else
            recon = _mm256_add_ps(recon, _mm256_add_ps(rlo_v, _mm256_mul_ps(rcode, rdelta_v)));
#endif
        }
        __m256 qv = _mm256_loadu_ps(qres + j);
        __m256 diff = _mm256_sub_ps(qv, recon);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(diff, diff, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
    }
    float dist = hsum256(acc);
#else
    uint64_t j = 0;
    float dist = 0.0f;
#endif

    for (; j < d; ++j) {
        float recon = lo + static_cast<float>(get_packed_code(primary, j, primary_bits)) * delta;
        if (use_residual) {
            recon += residual_lo
                + static_cast<float>(get_packed_code(residual, j, residual_bits)) * residual_delta;
        }
        float diff = qres[j] - recon;
        dist += diff * diff;
    }
    return dist;
}

template <uint64_t FixedD>
inline float lvq_distance_fixed(
    const uint8_t* rec,
    const float* qres,
    uint32_t primary_bits,
    uint32_t residual_bits,
    uint64_t primary_bytes,
    bool include_residual
) {
    return lvq_distance(
        rec,
        qres,
        FixedD,
        primary_bits,
        residual_bits,
        primary_bytes,
        include_residual);
}

inline float lvq_distance_dispatch(
    const uint8_t* rec,
    const float* qres,
    uint64_t d,
    uint32_t primary_bits,
    uint32_t residual_bits,
    uint64_t primary_bytes,
    bool include_residual
) {
    switch (d) {
        case 25: return lvq_distance_fixed<25>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 50: return lvq_distance_fixed<50>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 96: return lvq_distance_fixed<96>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 128: return lvq_distance_fixed<128>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 200: return lvq_distance_fixed<200>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 512: return lvq_distance_fixed<512>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 768: return lvq_distance_fixed<768>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 960: return lvq_distance_fixed<960>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        case 1024: return lvq_distance_fixed<1024>(rec, qres, primary_bits, residual_bits, primary_bytes, include_residual);
        default: return lvq_distance(rec, qres, d, primary_bits, residual_bits, primary_bytes, include_residual);
    }
}

class IVFLVQ {
public:
    uint64_t N = 0;
    uint64_t D = 0;
    uint64_t C = 0;
    uint32_t primary_bits = 8;
    uint32_t residual_bits = 0;
    uint64_t primary_bytes = 0;
    uint64_t residual_bytes = 0;
    uint64_t record_stride = 0;

    std::vector<uint64_t> start;
    std::vector<uint64_t> len;
    std::vector<uint32_t> id;
    std::vector<uint32_t> cluster_of_pos;
    std::vector<float> centroid;
    std::vector<float> residual_mean;
    std::vector<uint8_t> records;

    void configure(uint64_t n, uint64_t d, uint64_t c, uint32_t b1, uint32_t b2) {
        if (!((b1 == 4 || b1 == 8) && (b2 == 0 || b2 == 4 || b2 == 8))) {
            throw std::runtime_error("supported LVQ variants are B1 in {4,8}, B2 in {0,4,8}");
        }
        N = n;
        D = d;
        C = c;
        primary_bits = b1;
        residual_bits = b2;
        primary_bytes = packed_bytes(D, primary_bits);
        residual_bytes = residual_bits == 0 ? 0 : packed_bytes(D, residual_bits);
        record_stride = round_up(4 + primary_bytes + residual_bytes, 32);
    }

    void build_from(
        const float* xb,
        const float* centroids,
        const int* cluster_ids,
        uint64_t n,
        uint64_t d,
        uint64_t c,
        uint32_t b1,
        uint32_t b2
    ) {
        configure(n, d, c, b1, b2);
        centroid.assign(centroids, centroids + C * D);
        residual_mean.assign(D, 0.0f);
        start.assign(C, 0);
        len.assign(C, 0);

        for (uint64_t i = 0; i < N; ++i) {
            const int cid = cluster_ids[i];
            if (cid < 0 || static_cast<uint64_t>(cid) >= C) {
                throw std::runtime_error("cluster id out of range");
            }
            len[static_cast<uint64_t>(cid)]++;
        }
        uint64_t offset = 0;
        for (uint64_t ci = 0; ci < C; ++ci) {
            start[ci] = offset;
            offset += len[ci];
        }

        compute_residual_mean(xb, cluster_ids);

        id.assign(N, 0);
        cluster_of_pos.assign(N, 0);
        records.assign(N * record_stride, 0);

        std::vector<uint64_t> cursor(C, 0);
        std::vector<uint64_t> pos_of_row(N, 0);
        for (uint64_t i = 0; i < N; ++i) {
            const uint64_t cid = static_cast<uint64_t>(cluster_ids[i]);
            const uint64_t pos = start[cid] + cursor[cid]++;
            pos_of_row[i] = pos;
            id[pos] = static_cast<uint32_t>(i);
            cluster_of_pos[pos] = static_cast<uint32_t>(cid);
        }

#pragma omp parallel for schedule(static)
        for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(N); ++i_signed) {
            const uint64_t i = static_cast<uint64_t>(i_signed);
            const uint64_t cid = static_cast<uint64_t>(cluster_ids[i]);
            const uint64_t pos = pos_of_row[i];
            encode_one(xb + i * D, centroid.data() + cid * D, records.data() + pos * record_stride);
        }
        advise_huge_pages(records);
        advise_huge_pages(centroid);
    }

    ResultHeap search(
        const float* query,
        uint64_t k,
        uint64_t nprobe,
        double rerank_factor,
        SearchStats* stats = nullptr
    ) const {
        nprobe = std::min<uint64_t>(nprobe, C);
        if (k == 0 || nprobe == 0) return ResultHeap{};

        std::vector<Result> centroid_dist(C);
        for (uint64_t ci = 0; ci < C; ++ci) {
            centroid_dist[ci] = {sqr_l2_dispatch(query, centroid.data() + ci * D, D), static_cast<uint32_t>(ci)};
        }
        std::partial_sort(
            centroid_dist.begin(),
            centroid_dist.begin() + static_cast<std::ptrdiff_t>(nprobe),
            centroid_dist.end());

        const uint64_t candidate_k = residual_bits == 0
            ? k
            : std::max<uint64_t>(k, static_cast<uint64_t>(std::ceil(static_cast<double>(k) * rerank_factor)));
        FixedCandidateHeap first(candidate_k);
        std::vector<float> qres(D);

        for (uint64_t p = 0; p < nprobe; ++p) {
            const uint64_t ci = centroid_dist[p].second;
            make_query_residual(query, ci, qres.data());
            scan_list(ci, qres.data(), first, stats);
        }

        if (residual_bits == 0) {
            return first.to_result_heap();
        }

        std::vector<Candidate> candidates = first.sorted();
        FixedCandidateHeap refined(k);
        for (const Candidate& cand : candidates) {
            const uint64_t ci = cluster_of_pos[cand.pos];
            make_query_residual(query, ci, qres.data());
            const uint8_t* rec = records.data() + cand.pos * record_stride;
            const float dist = lvq_distance_dispatch(
                rec,
                qres.data(),
                D,
                primary_bits,
                residual_bits,
                primary_bytes,
                true);
            refined.push(dist, cand.id, cand.pos);
            if (stats) stats->reranked++;
        }
        return refined.to_result_heap();
    }

    void save(const char* path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error(std::string("cannot open for write: ") + path);
        const char magic[8] = {'L', 'V', 'Q', 'I', 'V', 'F', '0', '1'};
        const uint32_t version = 1;
        out.write(magic, sizeof(magic));
        write_scalar(out, version);
        write_scalar(out, N);
        write_scalar(out, D);
        write_scalar(out, C);
        write_scalar(out, primary_bits);
        write_scalar(out, residual_bits);
        write_scalar(out, primary_bytes);
        write_scalar(out, residual_bytes);
        write_scalar(out, record_stride);
        write_vector(out, start);
        write_vector(out, len);
        write_vector(out, id);
        write_vector(out, cluster_of_pos);
        write_vector(out, centroid);
        write_vector(out, residual_mean);
        write_vector(out, records);
    }

    void load(const char* path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error(std::string("cannot open: ") + path);
        char magic[8];
        in.read(magic, sizeof(magic));
        const char expected[8] = {'L', 'V', 'Q', 'I', 'V', 'F', '0', '1'};
        const char legacy_expected[8] = {
            static_cast<char>(0x4d), static_cast<char>(0x59), static_cast<char>(0x4c),
            static_cast<char>(0x56), static_cast<char>(0x51), static_cast<char>(0x49),
            static_cast<char>(0x56), static_cast<char>(0x46),
        };
        if (std::memcmp(magic, expected, sizeof(magic)) != 0 &&
            std::memcmp(magic, legacy_expected, sizeof(magic)) != 0) {
            throw std::runtime_error("not a LVQ IVF index");
        }
        uint32_t version = 0;
        read_scalar(in, version);
        if (version != 1) throw std::runtime_error("unsupported LVQ index version");
        read_scalar(in, N);
        read_scalar(in, D);
        read_scalar(in, C);
        read_scalar(in, primary_bits);
        read_scalar(in, residual_bits);
        read_scalar(in, primary_bytes);
        read_scalar(in, residual_bytes);
        read_scalar(in, record_stride);
        read_vector(in, start);
        read_vector(in, len);
        read_vector(in, id);
        read_vector(in, cluster_of_pos);
        read_vector(in, centroid);
        read_vector(in, residual_mean);
        read_vector(in, records);
        validate_loaded();
        advise_huge_pages(records);
        advise_huge_pages(centroid);
    }

private:
    void compute_residual_mean(const float* xb, const int* cluster_ids) {
        std::vector<double> sum(D, 0.0);
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        std::vector<double> local(static_cast<size_t>(nthreads) * D, 0.0);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* acc = local.data() + static_cast<size_t>(tid) * D;
#pragma omp for schedule(static)
            for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(N); ++i_signed) {
                const uint64_t i = static_cast<uint64_t>(i_signed);
                const uint64_t cid = static_cast<uint64_t>(cluster_ids[i]);
                const float* x = xb + i * D;
                const float* c = centroid.data() + cid * D;
                for (uint64_t j = 0; j < D; ++j) acc[j] += static_cast<double>(x[j] - c[j]);
            }
        }
        for (int t = 0; t < nthreads; ++t) {
            const double* acc = local.data() + static_cast<size_t>(t) * D;
            for (uint64_t j = 0; j < D; ++j) sum[j] += acc[j];
        }
#else
        for (uint64_t i = 0; i < N; ++i) {
            const uint64_t cid = static_cast<uint64_t>(cluster_ids[i]);
            const float* x = xb + i * D;
            const float* c = centroid.data() + cid * D;
            for (uint64_t j = 0; j < D; ++j) sum[j] += static_cast<double>(x[j] - c[j]);
        }
#endif
        for (uint64_t j = 0; j < D; ++j) residual_mean[j] = static_cast<float>(sum[j] / static_cast<double>(N));
    }

    void encode_one(const float* x, const float* c, uint8_t* rec) const {
        std::vector<float> centered(D);
        float lo = std::numeric_limits<float>::max();
        float hi = -std::numeric_limits<float>::max();
        for (uint64_t j = 0; j < D; ++j) {
            const float v = x[j] - c[j] - residual_mean[j];
            centered[j] = v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }

        store_half_pair(rec, lo, hi);
        uint8_t* primary = rec + 4;
        uint8_t* residual = primary + primary_bytes;
        std::memset(primary, 0, primary_bytes + residual_bytes);

        const uint32_t primary_levels = 1U << primary_bits;
        const float delta = (hi > lo) ? ((hi - lo) / static_cast<float>(primary_levels - 1U)) : 0.0f;
        const float inv_delta = delta > 0.0f ? 1.0f / delta : 0.0f;
        const uint32_t residual_levels = residual_bits > 0 ? (1U << residual_bits) : 0U;
        const float residual_lo = -0.5f * delta;
        const float residual_delta = (residual_bits > 0 && delta > 0.0f)
            ? (delta / static_cast<float>(residual_levels - 1U))
            : 0.0f;
        const float inv_residual_delta = residual_delta > 0.0f ? 1.0f / residual_delta : 0.0f;

        for (uint64_t j = 0; j < D; ++j) {
            int q = delta > 0.0f ? static_cast<int>(std::lrint((centered[j] - lo) * inv_delta)) : 0;
            q = std::max(0, std::min<int>(static_cast<int>(primary_levels) - 1, q));
            set_packed_code(primary, j, primary_bits, static_cast<uint8_t>(q));

            if (residual_bits > 0) {
                const float primary_decoded = lo + static_cast<float>(q) * delta;
                const float err = centered[j] - primary_decoded;
                int rq = residual_delta > 0.0f
                    ? static_cast<int>(std::lrint((err - residual_lo) * inv_residual_delta))
                    : 0;
                rq = std::max(0, std::min<int>(static_cast<int>(residual_levels) - 1, rq));
                set_packed_code(residual, j, residual_bits, static_cast<uint8_t>(rq));
            }
        }
    }

    void make_query_residual(const float* query, uint64_t ci, float* qres) const {
        const float* c = centroid.data() + ci * D;
        for (uint64_t j = 0; j < D; ++j) qres[j] = query[j] - c[j] - residual_mean[j];
    }

    void scan_list(uint64_t ci, const float* qres, FixedCandidateHeap& heap, SearchStats* stats) const {
        switch (D) {
            case 25: scan_list_fixed<25>(ci, qres, heap, stats); return;
            case 50: scan_list_fixed<50>(ci, qres, heap, stats); return;
            case 96: scan_list_fixed<96>(ci, qres, heap, stats); return;
            case 128: scan_list_fixed<128>(ci, qres, heap, stats); return;
            case 200: scan_list_fixed<200>(ci, qres, heap, stats); return;
            case 512: scan_list_fixed<512>(ci, qres, heap, stats); return;
            case 768: scan_list_fixed<768>(ci, qres, heap, stats); return;
            case 960: scan_list_fixed<960>(ci, qres, heap, stats); return;
            case 1024: scan_list_fixed<1024>(ci, qres, heap, stats); return;
            default: scan_list_runtime(ci, qres, heap, stats); return;
        }
    }

    template <uint64_t FixedD>
    void scan_list_fixed(uint64_t ci, const float* qres, FixedCandidateHeap& heap, SearchStats* stats) const {
        const uint64_t begin = start[ci];
        const uint64_t end = begin + len[ci];
        constexpr uint64_t prefetch_offset = 8;
        for (uint64_t pos = begin; pos < end; ++pos) {
            const uint64_t pf = pos + prefetch_offset;
            if (pf < end) {
                _mm_prefetch(reinterpret_cast<const char*>(records.data() + pf * record_stride), _MM_HINT_T0);
            }
            const uint8_t* rec = records.data() + pos * record_stride;
            const float dist = lvq_distance_fixed<FixedD>(
                rec,
                qres,
                primary_bits,
                residual_bits,
                primary_bytes,
                false);
            heap.push(dist, id[pos], pos);
            if (stats) stats->scanned++;
        }
    }

    void scan_list_runtime(uint64_t ci, const float* qres, FixedCandidateHeap& heap, SearchStats* stats) const {
        const uint64_t begin = start[ci];
        const uint64_t end = begin + len[ci];
        constexpr uint64_t prefetch_offset = 8;
        for (uint64_t pos = begin; pos < end; ++pos) {
            const uint64_t pf = pos + prefetch_offset;
            if (pf < end) {
                _mm_prefetch(reinterpret_cast<const char*>(records.data() + pf * record_stride), _MM_HINT_T0);
            }
            const uint8_t* rec = records.data() + pos * record_stride;
            const float dist = lvq_distance(rec, qres, D, primary_bits, residual_bits, primary_bytes, false);
            heap.push(dist, id[pos], pos);
            if (stats) stats->scanned++;
        }
    }

    template <class T>
    static void write_scalar(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <class T>
    static void read_scalar(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) throw std::runtime_error("short read while loading LVQ index");
    }

    template <class T>
    static void write_vector(std::ofstream& out, const std::vector<T>& vec) {
        uint64_t n = static_cast<uint64_t>(vec.size());
        write_scalar(out, n);
        if (n) out.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(n * sizeof(T)));
    }

    template <class T>
    static void read_vector(std::ifstream& in, std::vector<T>& vec) {
        uint64_t n = 0;
        read_scalar(in, n);
        vec.resize(n);
        if (n) {
            in.read(reinterpret_cast<char*>(vec.data()), static_cast<std::streamsize>(n * sizeof(T)));
            if (!in) throw std::runtime_error("short vector read while loading LVQ index");
        }
    }

    void validate_loaded() const {
        if (start.size() != C || len.size() != C) throw std::runtime_error("invalid list metadata");
        if (id.size() != N || cluster_of_pos.size() != N) throw std::runtime_error("invalid id metadata");
        if (centroid.size() != C * D) throw std::runtime_error("invalid centroid metadata");
        if (residual_mean.size() != D) throw std::runtime_error("invalid residual mean metadata");
        if (records.size() != N * record_stride) throw std::runtime_error("invalid record storage");
    }
};

inline void parse_compression(const std::string& name, uint32_t& b1, uint32_t& b2) {
    if (name == "LVQ8x0") {
        b1 = 8; b2 = 0; return;
    }
    if (name == "LVQ4x0") {
        b1 = 4; b2 = 0; return;
    }
    if (name == "LVQ4x4") {
        b1 = 4; b2 = 4; return;
    }
    if (name == "LVQ4x8") {
        b1 = 4; b2 = 8; return;
    }
    if (name == "LVQ8x8") {
        b1 = 8; b2 = 8; return;
    }
    throw std::runtime_error("unsupported compression: " + name);
}

inline double bits_per_dim(uint32_t b1, uint32_t b2) {
    return static_cast<double>(b1 + b2);
}

}  // namespace lvq
