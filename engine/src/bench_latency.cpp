#include "features.hpp"
#include "model_scorer.hpp"
#include "timing.hpp"
#include <iostream>

int main(){
    FeatureEngine fe(5); fe.prepare();
    LinearLogitScorer scorer; scorer.load_weights("engine/src/onnx/linear.txt");

    L2Tick t{};
    for(int i=0;i<5;i++){ t.bid_px[i]=100.f-0.01f*i; t.ask_px[i]=100.01f+0.01f*i; t.bid_sz[i]=1000; t.ask_sz[i]=900; }
    FeatureVec fx; fx.x.reserve(64);

    for(int i=0;i<10000;i++){ fe.compute(t, fx); (void)scorer.score(fx.x); } // warmup

    const int N=200000;
    double sum=0, maxv=0;
    Timer tm;
    for(int i=0;i<N;i++){
        tm.start();
        fe.compute(t, fx);
        float p = scorer.score(fx.x);
        double e2e = tm.us();
        sum += e2e; if(e2e>maxv) maxv=e2e;
        (void)p;
    }
    std::cout << "mean_us=" << (sum/N) << " max_us=" << maxv << "\n";
    return 0;
}
