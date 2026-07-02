// lvq_search
// Run an nprobe sweep over an LVQ IVF index.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ivf_lvq.h"
#include "matrix.h"

static float compute_recall(ResultHeap raw, const int* gt, int qi, int k, int gt_k) {
    int correct = 0;
    while (!raw.empty()) {
        const uint32_t rid = raw.top().second;
        raw.pop();
        for (int j = 0; j < k && j < gt_k; ++j) {
            if (static_cast<int>(rid) == gt[qi * gt_k + j]) {
                ++correct;
                break;
            }
        }
    }
    return static_cast<float>(correct) / static_cast<float>(std::min(k, gt_k));
}

static void parse_nprobes(const char* value, std::vector<int>& out) {
    std::stringstream ss(value);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        int v = std::stoi(tok);
        if (v > 0) out.push_back(v);
    }
}

static void usage() {
    std::cerr
        << "Usage: lvq_search -i <index> -q <query.fvecs> -g <gt.ivecs> "
        << "-k <top_k> -p <nprobe_csv> [-w warmup] [-m measure] "
        << "[--rerank-factor value]\n";
}

int main(int argc, char** argv) {
    const struct option longopts[] = {
        {"index", required_argument, 0, 'i'},
        {"query", required_argument, 0, 'q'},
        {"groundtruth", required_argument, 0, 'g'},
        {"topk", required_argument, 0, 'k'},
        {"probes", required_argument, 0, 'p'},
        {"warmup", required_argument, 0, 'w'},
        {"measure", required_argument, 0, 'm'},
        {"rerank-factor", required_argument, 0, 1000},
        {nullptr, 0, nullptr, 0},
    };

    std::string index_path;
    std::string query_path;
    std::string gt_path;
    int k = 10;
    int warmup = 0;
    int measure = 2;
    double rerank_factor = 10.0;
    std::vector<int> nprobes;

    int opt, ind;
    while ((opt = getopt_long(argc, argv, "i:q:g:k:p:w:m:", longopts, &ind)) != -1) {
        switch (opt) {
            case 'i': index_path = optarg; break;
            case 'q': query_path = optarg; break;
            case 'g': gt_path = optarg; break;
            case 'k': k = std::atoi(optarg); break;
            case 'p': parse_nprobes(optarg, nprobes); break;
            case 'w': warmup = std::atoi(optarg); break;
            case 'm': measure = std::atoi(optarg); break;
            case 1000: rerank_factor = std::atof(optarg); break;
            default: usage(); return 1;
        }
    }

    if (index_path.empty() || query_path.empty() || gt_path.empty() || k <= 0 || nprobes.empty()) {
        usage();
        return 1;
    }
    warmup = std::max(0, warmup);
    measure = std::max(1, measure);
    rerank_factor = std::max(1.0, rerank_factor);

    try {
        lvq::IVFLVQ ivf;
        ivf.load(index_path.c_str());
        Matrix<float> queries(const_cast<char*>(query_path.c_str()));
        Matrix<int> gt(const_cast<char*>(gt_path.c_str()));

        if (queries.d != ivf.D) {
            std::cerr << "query dimension mismatch: " << queries.d << " vs " << ivf.D << "\n";
            return 1;
        }
        if (gt.n != queries.n) {
            std::cerr << "groundtruth/query cardinality mismatch\n";
            return 1;
        }

        std::cerr << "LVQ search: N=" << ivf.N
                  << " D=" << ivf.D
                  << " C=" << ivf.C
                  << " B1=" << ivf.primary_bits
                  << " B2=" << ivf.residual_bits
                  << " nq=" << queries.n
                  << " rerank_factor=" << rerank_factor << "\n";

        for (int nprobe_in : nprobes) {
            const uint64_t nprobe = std::min<uint64_t>(static_cast<uint64_t>(nprobe_in), ivf.C);
            for (int w = 0; w < warmup; ++w) {
                for (uint64_t qi = 0; qi < queries.n; ++qi) {
                    ivf.search(queries.data + qi * ivf.D, static_cast<uint64_t>(k), nprobe, rerank_factor);
                }
            }

            std::vector<double> elapsed;
            elapsed.reserve(static_cast<size_t>(measure));
            float recall = 0.0f;
            lvq::SearchStats stats;

            for (int pass = 0; pass < measure; ++pass) {
                lvq::SearchStats pass_stats;
                float pass_recall = 0.0f;
                auto t0 = std::chrono::steady_clock::now();
                for (uint64_t qi = 0; qi < queries.n; ++qi) {
                    lvq::SearchStats q_stats;
                    ResultHeap res = ivf.search(
                        queries.data + qi * ivf.D,
                        static_cast<uint64_t>(k),
                        nprobe,
                        rerank_factor,
                        &q_stats);
                    pass_stats.add(q_stats);
                    pass_recall += compute_recall(
                        res,
                        gt.data,
                        static_cast<int>(qi),
                        k,
                        static_cast<int>(gt.d));
                }
                auto t1 = std::chrono::steady_clock::now();
                elapsed.push_back(std::chrono::duration<double>(t1 - t0).count());
                if (pass == measure - 1) {
                    recall = pass_recall / static_cast<float>(queries.n);
                    stats = pass_stats;
                }
            }

            std::sort(elapsed.begin(), elapsed.end());
            const double med = elapsed[elapsed.size() / 2];
            const double qps = static_cast<double>(queries.n) / std::max(med, 1e-12);
            std::cout << "RESULT\t" << nprobe
                      << "\t" << recall
                      << "\t" << qps
                      << "\t" << stats.scanned
                      << "\t" << stats.reranked
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "lvq_search failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
