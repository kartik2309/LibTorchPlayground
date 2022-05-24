cmake_minimum_required(VERSION 3.22)
project(LibTorchPlayground)

# Additional Packages
find_package (Python3 COMPONENTS Interpreter Development)

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(TorchVision REQUIRED)

find_package(RapidJSON REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RapidJSON_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)

add_executable(
        LibTorchPlayground

        main.cpp

        CIFAR/Models/BlockConvNet/BlockConvNet.cpp
        CIFAR/Models/BlockConvNet/BlockConvNet.h

        CIFAR/Dataset/CIFARTorchDataset.cpp
        CIFAR/Dataset/CIFARTorchDataset.h

        CIFAR/Trainer/Trainer.cpp
        CIFAR/Trainer/Trainer.h CIFAR/Trainer/TrainerImpl.cpp)

# Link the libraries
target_link_libraries(LibTorchPlayground ${RapidJSON_LIBRARIES})
target_link_libraries(LibTorchPlayground "${OpenCV_LIBRARIES}")
target_link_libraries(LibTorchPlayground "${TORCH_LIBRARIES}")
target_link_libraries(LibTorchPlayground Python3::Python)
target_link_libraries(LibTorchPlayground TorchVision::TorchVision)


