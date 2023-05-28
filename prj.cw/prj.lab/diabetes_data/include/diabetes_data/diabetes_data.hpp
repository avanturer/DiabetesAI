//
// Created by hedge on 13.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023DDAT
#define DIABETES_AI_HPP_09052023DDAT

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

    explicit DiabetesData(const string &file_path, const string &data_name);

    void load_data_from_file(const string &file_path, const string &data_name);

    explicit DiabetesData(const vector<vector<double>> &features);

    static vector<vector<double>> data_normalization(vector<vector<double>> X);

    vector<vector<double>> get_X ();

    vector<int> get_y ();

private:
    vector<vector<double>> features_;
    vector<vector<string>> dataset_;
    vector<vector<double>> X_;
    vector<int> y_;
};

#endif
