#pragma once
#include <atomic>
#include <vector>
#include <cstddef>

template<typename T>
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(size_t capacity)
        : capacity_(capacity), mask_(capacity-1), buf_(capacity) {}
    bool push(const T& v) {
        auto h = head_.load(std::memory_order_relaxed);
        auto n = (h + 1) & mask_;
        if (n == tail_.load(std::memory_order_acquire)) return false;
        buf_[h] = v;
        head_.store(n, std::memory_order_release);
        return true;
    }
    bool pop(T& out) {
        auto t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        out = buf_[t];
        tail_.store((t + 1) & mask_, std::memory_order_release);
        return true;
    }
private:
    size_t capacity_, mask_;
    std::vector<T> buf_;
    std::atomic<size_t> head_{0}, tail_{0};
};
