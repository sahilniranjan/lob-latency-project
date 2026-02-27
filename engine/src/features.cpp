#include "features.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

FeatureEngine::FeatureEngine(int levels)
    : L(levels),
      roll_ofi_     {RollingMean(ROLLING_TICKS[0]), RollingMean(ROLLING_TICKS[1]), RollingMean(ROLLING_TICKS[2])},
      roll_spread_  {RollingMean(ROLLING_TICKS[0]), RollingMean(ROLLING_TICKS[1]), RollingMean(ROLLING_TICKS[2])},
      roll_qi0_     {RollingMean(ROLLING_TICKS[0]), RollingMean(ROLLING_TICKS[1]), RollingMean(ROLLING_TICKS[2])}
{}

void FeatureEngine::prepare(size_t) {}

int FeatureEngine::feature_dim() const {
    // mid + spread + ofi + microprice + L*qi + 3 windows * 3 signals
    return 4 + L + N_ROLLING_WINDOWS * 3;
}

void FeatureEngine::compute(const L2Tick& t, FeatureVec& out) {
    out.x.clear();

    // --- Canonical feature order (must match Python training) ---
    // [mid, spread, ofi, microprice,
    //  qi_0 .. qi_{L-1},
    //  rolling_ofi_w0, rolling_ofi_w1, rolling_ofi_w2,
    //  rolling_spread_w0, rolling_spread_w1, rolling_spread_w2,
    //  rolling_qi0_w0, rolling_qi0_w1, rolling_qi0_w2]

    float mid = 0.5f * (t.ask_px[0] + t.bid_px[0]);
    float spread = t.ask_px[0] - t.bid_px[0];

    float ofi = (t.bid_sz[0] - prev_bid_sz0) - (t.ask_sz[0] - prev_ask_sz0);
    prev_bid_sz0 = t.bid_sz[0];
    prev_ask_sz0 = t.ask_sz[0];

    float microprice = (t.bid_px[0] * t.ask_sz[0] + t.ask_px[0] * t.bid_sz[0])
                     / (t.bid_sz[0] + t.ask_sz[0] + 1e-6f);

    out.x.push_back(mid);
    out.x.push_back(spread);
    out.x.push_back(ofi);
    out.x.push_back(microprice);

    // Queue imbalance per level
    float qi0 = 0.f;
    for (int i = 0; i < L; i++) {
        float qi = (t.bid_sz[i] - t.ask_sz[i]) / (t.bid_sz[i] + t.ask_sz[i] + 1e-6f);
        out.x.push_back(qi);
        if (i == 0) qi0 = qi;
    }

    // Rolling window features
    for (int w = 0; w < N_ROLLING_WINDOWS; w++)
        out.x.push_back(roll_ofi_[w].update(ofi));
    for (int w = 0; w < N_ROLLING_WINDOWS; w++)
        out.x.push_back(roll_spread_[w].update(spread));
    for (int w = 0; w < N_ROLLING_WINDOWS; w++)
        out.x.push_back(roll_qi0_[w].update(qi0));
}

// ── CSV tick reader ──────────────────────────────────────────────
// Expects header row with columns: ts,bid_px_1,...,ask_sz_5
std::vector<L2Tick> load_ticks_csv(const std::string& path, int levels) {
    std::vector<L2Tick> ticks;
    std::ifstream f(path);
    if (!f.is_open()) return ticks;

    std::string line;
    // Read and parse header to find column indices
    std::getline(f, line);  // skip header

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string cell;
        std::vector<float> vals;

        while (std::getline(ss, cell, ',')) {
            try { vals.push_back(std::stof(cell)); }
            catch (...) { vals.push_back(0.f); }
        }

        // Minimum: ts + 4 fields × levels
        if (static_cast<int>(vals.size()) < 1 + 4 * levels) continue;

        L2Tick t{};
        t.ts_ns = static_cast<uint64_t>(vals[0]);
        for (int i = 0; i < std::min(levels, MAX_LEVELS); i++) {
            t.bid_px[i] = vals[1 + i];
            t.ask_px[i] = vals[1 + levels + i];
            t.bid_sz[i] = vals[1 + 2 * levels + i];
            t.ask_sz[i] = vals[1 + 3 * levels + i];
        }
        ticks.push_back(t);
    }
    return ticks;
}
