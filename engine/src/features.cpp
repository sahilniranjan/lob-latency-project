#include "features.hpp"

FeatureEngine::FeatureEngine(int levels): L(levels) {}
void FeatureEngine::prepare(size_t) {}

void FeatureEngine::compute(const L2Tick& t, FeatureVec& out) {
    out.x.clear();
    float mid = 0.5f * (t.ask_px[0] + t.bid_px[0]);
    float spread = t.ask_px[0] - t.bid_px[0];

    for(int i=0;i<L;i++){
        float qi = (t.bid_sz[i]-t.ask_sz[i])/(t.bid_sz[i]+t.ask_sz[i]+1e-6f);
        out.x.push_back(qi);
    }
    float ofi = (t.bid_sz[0]-prev_bid_sz0) - (t.ask_sz[0]-prev_ask_sz0);
    prev_bid_sz0 = t.bid_sz[0]; prev_ask_sz0 = t.ask_sz[0];

    out.x.push_back(mid);
    out.x.push_back(spread);
    out.x.push_back(ofi);
}
