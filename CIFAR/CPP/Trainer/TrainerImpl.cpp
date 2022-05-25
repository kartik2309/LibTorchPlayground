//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#include "../Dataset/CIFARTorchDataset.h"
#include "../Models/BlockConvNet/BlockConvNet.h"
#include "Trainer.cpp"

template class Trainer<BlockConvNet *,
                       CIFARTorchDataset,
                       torch::data::transforms::Stack<>,
                       torch::data::samplers::SequentialSampler,
                       torch::optim::AdamW,
                       torch::nn::CrossEntropyLoss>;