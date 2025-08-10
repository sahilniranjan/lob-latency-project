#pragma once
#include <chrono>
struct Timer {
    using clk = std::chrono::steady_clock;
    clk::time_point t0;
    void start(){ t0=clk::now(); }
    double us() const { return std::chrono::duration<double,std::micro>(clk::now()-t0).count(); }
};
