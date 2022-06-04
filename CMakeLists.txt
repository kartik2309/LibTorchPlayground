cmake_minimum_required(VERSION 3.22)
project(LibTorchPlayground)

# Additional Packages
find_package(Python3 COMPONENTS Interpreter Development)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(TorchVision REQUIRED)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h)
find_library(ONNX_RUNTIME_LIB onnxruntime)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()

find_package(Boost COMPONENTS program_options log log_setup mpi serialization REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DBOOST_LOG_DYN_LINK}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_executable(
        LibTorchPlayground

        main.cpp

        Projects/CIFAR/CPP/Models/BlockConvNet/BlockConvNet.cpp
        Projects/CIFAR/CPP/Models/BlockConvNet/BlockConvNet.h

        Projects/CIFAR/CPP/Dataset/CIFARTorchDataset.cpp
        Projects/CIFAR/CPP/Dataset/CIFARTorchDataset.h

        Projects/CIFAR/CPP/cifar.hpp

        TorchScriptUtilities/CPP/LoadJIT/LoadJIT.cpp
        TorchScriptUtilities/CPP/LoadJIT/LoadJIT.h

        Trainer/CPP/Trainer.hpp

        Trainer/CPP/Utilities/BatchTransformTemplate.hpp

        ONNX/Inference/Inference.cpp
        ONNX/Inference/Inference.h
        Trainer/CPP/Utilities/MultiprocessingBackend.hpp Trainer/CPP/Utilities/Metrics/Metrics.cpp Trainer/CPP/Utilities/Metrics/Metrics.h)

# Link the libraries
#target_link_libraries(LibTorchPlayground stdc++fs)
target_link_libraries(LibTorchPlayground "${OpenCV_LIBRARIES}")
target_link_libraries(LibTorchPlayground "${TORCH_LIBRARIES}")
target_link_libraries(LibTorchPlayground Python3::Python)
target_link_libraries(LibTorchPlayground TorchVision::TorchVision)
target_link_libraries(LibTorchPlayground ${Boost_LOG_LIBRARY})
target_link_libraries(LibTorchPlayground ${Boost_LOG_SETUP_LIBRARY})
target_link_libraries(LibTorchPlayground ${Boost_LIBRARIES})
target_link_libraries(LibTorchPlayground OpenMP::OpenMP_CXX)
target_link_libraries(LibTorchPlayground ${ONNX_RUNTIME_LIB})


