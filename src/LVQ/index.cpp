// lvq_index
// Build an IVF residual LVQ index from Faiss-generated centroids/assignments.

#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <string>

#include "ivf_lvq.h"
#include "matrix.h"

static void usage() {
    std::cerr
        << "Usage: lvq_index -s <source/> -c <nlist> "
        << "--compression <LVQ8x0|LVQ4x0|LVQ4x4|LVQ4x8> -o <index>\n";
}

int main(int argc, char** argv) {
    const struct option longopts[] = {
        {"source", required_argument, 0, 's'},
        {"clusters", required_argument, 0, 'c'},
        {"compression", required_argument, 0, 'z'},
        {"output", required_argument, 0, 'o'},
        {nullptr, 0, nullptr, 0},
    };

    std::string source;
    std::string compression = "LVQ8x0";
    std::string output;
    int nlist = 4096;

    int opt, ind;
    while ((opt = getopt_long(argc, argv, "s:c:z:o:", longopts, &ind)) != -1) {
        switch (opt) {
            case 's': source = optarg; break;
            case 'c': nlist = std::atoi(optarg); break;
            case 'z': compression = optarg; break;
            case 'o': output = optarg; break;
            default: usage(); return 1;
        }
    }

    if (source.empty() || output.empty() || nlist <= 0) {
        usage();
        return 1;
    }
    if (source.back() != '/') source.push_back('/');

    uint32_t b1 = 0;
    uint32_t b2 = 0;
    try {
        lvq::parse_compression(compression, b1, b2);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const std::string base_path = source + "base.fvecs";
    const std::string centroid_path = source + "centroids.fvecs";
    const std::string cluster_path = source + "cluster_ids.ivecs";

    Matrix<float> base(const_cast<char*>(base_path.c_str()));
    Matrix<float> centroids(const_cast<char*>(centroid_path.c_str()));
    Matrix<int> cluster_ids(const_cast<char*>(cluster_path.c_str()));

    if (centroids.n != static_cast<size_t>(nlist) || centroids.d != base.d) {
        std::cerr << "centroid shape mismatch\n";
        return 1;
    }
    if (cluster_ids.n != base.n || cluster_ids.d != 1) {
        std::cerr << "cluster id shape mismatch\n";
        return 1;
    }

    try {
        lvq::IVFLVQ ivf;
        ivf.build_from(
            base.data,
            centroids.data,
            cluster_ids.data,
            static_cast<uint64_t>(base.n),
            static_cast<uint64_t>(base.d),
            static_cast<uint64_t>(nlist),
            b1,
            b2);
        ivf.save(output.c_str());
        std::cerr << "LVQ index saved: N=" << ivf.N
                  << " D=" << ivf.D
                  << " C=" << ivf.C
                  << " compression=" << compression
                  << " stride=" << ivf.record_stride
                  << " bytes=" << ivf.records.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "lvq_index failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
