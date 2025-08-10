#include "model_scorer.hpp"
#include <cmath>
#include <fstream>

static inline float sigmoid(float z){ return 1.f/(1.f+std::exp(-z)); }

float LinearLogitScorer::score(const std::vector<float>& x){
    float z=b;
    size_t n = x.size() < w.size() ? x.size() : w.size();
    for(size_t i=0;i<n;++i) z += w[i]*x[i];
    return sigmoid(z);
}

bool LinearLogitScorer::load_weights(const std::string& path){
    std::ifstream f(path);
    if(!f) return false;
    size_t n; f>>b; f>>n; w.resize(n);
    for(size_t i=0;i<n;++i) f>>w[i];
    return true;
}
