#pragma once
#include <array>
#include <cstdint>
#include <vector>

struct L2Tick {
    uint64_t ts_ns; // timestamp
    std::array<float,5> bid_px, ask_px, bid_sz, ask_sz; // L=5 example
};

struct FeatureVec {
    std::vector<float> x;
};

class FeatureEngine {
public:
    explicit FeatureEngine(int levels=5);
    void prepare(size_t max_batch=1);
    void compute(const L2Tick& t, FeatureVec& out);
private:
    int L;
    float prev_bid_sz0{0}, prev_ask_sz0{0};
};
