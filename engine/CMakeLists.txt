cmake_minimum_required(VERSION 3.16)
project(lob_engine CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ONNXRUNTIME "Use ONNX Runtime for inference" ON)

find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(replay src/main_replay.cpp src/features.cpp src/model_scorer.cpp)
add_executable(bench src/bench_latency.cpp src/features.cpp src/model_scorer.cpp)

target_link_libraries(replay PRIVATE Threads::Threads)
target_link_libraries(bench PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(replay PRIVATE /O2)
  target_compile_options(bench PRIVATE /O2)
else()
  target_compile_options(replay PRIVATE -O3 -march=native -flto)
  target_compile_options(bench PRIVATE -O3 -march=native -flto)
endif()
