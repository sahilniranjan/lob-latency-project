#pragma once
#include <vector>
#include <string>

class ModelScorer {
public:
    virtual ~ModelScorer() = default;
    virtual float score(const std::vector<float>& x) = 0;
};

class LinearLogitScorer : public ModelScorer {
public:
    bool load_weights(const std::string& path);
    float score(const std::vector<float>& x) override;
private:
    std::vector<float> w; float b{0};
};
