#include "ring_buffer.hpp"
#include "features.hpp"
#include "model_scorer.hpp"
#include "timing.hpp"
#include <thread>
#include <atomic>
#include <iostream>

struct TickMsg { L2Tick t; };

int main(){
    SpscRingBuffer<TickMsg> q(1<<17);
    FeatureEngine fe(5); fe.prepare();
    LinearLogitScorer scorer; scorer.load_weights("engine/src/onnx/linear.txt");

    std::atomic<bool> run{true};

    std::thread prod([&]{
        for(int i=0;i<200000;i++){
            TickMsg m{};
            for(int k=0;k<5;k++){ m.t.bid_px[k]=100-0.01*k; m.t.ask_px[k]=100.01+0.01*k; m.t.bid_sz[k]=1000; m.t.ask_sz[k]=900; }
            while(!q.push(m)) { }
        }
        run=false;
    });

    std::thread worker([&]{
        TickMsg m; FeatureVec fx; fx.x.reserve(64);
        Timer tm; size_t cnt=0; double sum=0, maxv=0;
        while(run || q.pop(m)){
            if(!q.pop(m)) continue;
            tm.start();
            fe.compute(m.t, fx);
            float p = scorer.score(fx.x);
            double us = tm.us();
            sum += us; if(us>maxv) maxv=us; cnt++;
            (void)p;
        }
        std::cout << "worker_mean_us=" << (sum/(cnt+1e-9)) << " max_us=" << maxv << "\n";
    });

    prod.join(); worker.join();
    return 0;
}
