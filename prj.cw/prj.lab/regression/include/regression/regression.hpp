//
// Created by hedge on 09.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023REG
#define DIABETES_AI_HPP_09052023REG

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

class LogisticRegression {
public:
    LogisticRegression(const vector<vector<double>> &X, const vector<int> &y);

    static vector<vector<double>> logit(vector<vector<double>> X, vector<double> w);

    static vector<vector<double>> sigmoid(vector<vector<double>> logits);

    vector<double> fit(int max_iter = 100, double lr = 0.1);

    double loss(vector<int> y, vector<vector<double>> z);

    static void save_weights(const vector<double> &weights);

    static vector<vector<double>> predict_proba(vector<vector<double>> feauters);

    static vector<int> predict(const vector<vector<double>> &feauters, double threshold = 0.5);

    static int model_accuracy(vector<int> results, vector<int> y);

    static void saveLossToCSV(const vector<double> &losses);

private:
    vector<vector<double>> X_;
    vector<int> y_;
    vector<double> w_;
    vector<double> losses_;
    vector<vector<double>> logloss_;
    int max_iter_ = 0;
    double lr_ = 0;
};

#endif
