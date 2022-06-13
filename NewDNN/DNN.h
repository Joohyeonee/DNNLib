#pragma once

#include "NN.h"

__declspec(dllexport) void setInput(NeuralNet &NN);
__declspec(dllexport) void setInput(NeuralNet &NN, int startRow, int EndRow);
__declspec(dllexport) void setTarget(NeuralNet &NN);
__declspec(dllexport) void setTarget(NeuralNet &NN, int startRow, int EndRow);
__declspec(dllexport) void normalization(NeuralNet &NN);
__declspec(dllexport) void denormalization(NeuralNet &NN);
__declspec(dllexport) void feedForward(NeuralNet &NN);
__declspec(dllexport) void backPropagation(NeuralNet &NN);
__declspec(dllexport) void predict(NeuralNet &NN);
__declspec(dllexport) void GDMoment(NeuralNet &NN, int layer);
__declspec(dllexport) void RMSProp(NeuralNet &NN, int layer);
__declspec(dllexport) void Adagrad(NeuralNet &NN, int layer);
__declspec(dllexport) void Adam(NeuralNet &NN, int layer);
__declspec(dllexport) void updateWeight(NeuralNet &NN);
__declspec(dllexport) void makeModel(NeuralNet &model, NeuralNet &NN);

__declspec(dllexport) double linearF(double x);
__declspec(dllexport) double dlinearF(double x);
__declspec(dllexport) double sigmoidF(double x);
__declspec(dllexport) double dsigmoidF(double x);
__declspec(dllexport) double reluF(double x);
__declspec(dllexport) double dreluF(double x);
__declspec(dllexport) double softMaxF(double x, double sum);
__declspec(dllexport) double dsoftMaxF(double x);
__declspec(dllexport) double activeF(double x, int act, double sum);
__declspec(dllexport) double dactiveF(double x, int act, double sum);

__declspec(dllexport) double calcRMSE(NeuralNet &NN);