//
// Created by hedge on 22.04.2023.
//
#include "ai_realization.hpp"
#include "plotting.hpp"
#include "matplotlibcpp.h"
int main() {
    DiabetesData a("dataset1");
    vector<vector<double>> X1 = a.X;
    vector<int> y1 = a.y;
//    LogisticRegression lg1(X1, y1);
//    vector<double> losses = lg1.fit();
//    for (double losse: losses)
//        cout << losse << endl;

//    DiabetesData b("dataset2");
//    vector<vector<double>> X = b.X;
//    vector<int> y = b.y;
//    LogisticRegression lg2(X, y);
//    vector<int> results = LogisticRegression::predict(X);
//    cout << to_string(LogisticRegression::model_accuracy(results, b.y)) + "%";

}