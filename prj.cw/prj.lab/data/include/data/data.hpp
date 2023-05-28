//
// Created by hedge on 13.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023
#define DIABETES_AI_HPP_09052023

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>


using namespace std;

class DiabetesData {
public:

    explicit DiabetesData(const string &data_name);

    void load_data_from_file(const string &data_name);

    explicit DiabetesData(const vector<vector<double>> &features);

    static vector<vector<double>> data_normalization(vector<vector<double>> X);

private:
    vector<vector<double>> features_;
    vector<vector<string>> dataset_;
    vector<vector<double>> X_;
    vector<int> y_;

};

#endif
