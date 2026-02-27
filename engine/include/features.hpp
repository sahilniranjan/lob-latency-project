#pragma once
#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

constexpr int MAX_LEVELS = 5;
constexpr int N_ROLLING_WINDOWS = 3;              // 10, 20, 50 ms
constexpr int ROLLING_TICKS[] = {1, 2, 5};         // tick approximations

struct L2Tick {
    uint64_t ts_ns;  // timestamp in nanoseconds
    std::array<float, MAX_LEVELS> bid_px, ask_px, bid_sz, ask_sz;
};

struct FeatureVec {
    std::vector<float> x;
};

// Simple fixed-capacity circular buffer for rolling means
class RollingMean {
public:
    explicit RollingMean(int window = 1)
        : win_(window), buf_(window, 0.0f), sum_(0.0f), idx_(0), filled_(false) {}

    float update(float val) {
        sum_ -= buf_[idx_];
        buf_[idx_] = val;
        sum_ += val;
        idx_ = (idx_ + 1) % win_;
        if (!filled_ && idx_ == 0) filled_ = true;
        int n = filled_ ? win_ : idx_;
        return (n > 0) ? sum_ / static_cast<float>(n) : val;
    }
private:
    int win_;
    std::vector<float> buf_;
    float sum_;
    int idx_;
    bool filled_;
};

class FeatureEngine {
public:
    explicit FeatureEngine(int levels = 5);
    void prepare(size_t max_batch = 1);
    void compute(const L2Tick& t, FeatureVec& out);
    int feature_dim() const;
private:
    int L;
    float prev_bid_sz0{0}, prev_ask_sz0{0};
    // Rolling windows for OFI, spread, top-level imbalance
    std::array<RollingMean, N_ROLLING_WINDOWS> roll_ofi_;
    std::array<RollingMean, N_ROLLING_WINDOWS> roll_spread_;
    std::array<RollingMean, N_ROLLING_WINDOWS> roll_qi0_;
};

// ── CSV tick reader (for replaying real data) ────────────────────
std::vector<L2Tick> load_ticks_csv(const std::string& path, int levels = 5);
