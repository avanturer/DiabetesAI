//
// Created by hedge on 09.05.2023.
//
#pragma once
#ifndef DIABETES_AI_HPP_09052023PLOT
#define DIABETES_AI_HPP_09052023PLOT

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

class Plot {
public:
    Plot() = default;
    static void CreateLatexFile(vector<int> results, vector<int> y, int max_iter = 100);
private:
};

#endif
