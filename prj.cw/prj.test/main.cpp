//
// Created by hedge on 22.04.2023.
//
#include <regression/regression.hpp>
#include <diabetes_data/diabetes_data.hpp>
#include <plot/plot.hpp>

int main() {
    DiabetesData a("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data","dataset1");
    vector<vector<double>> X1 = a.get_X();
    vector<int> y1 = a.get_y();
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit(10);
    for (double losse: losses)
        cout << losse << endl;

//    DiabetesData b("dataset2");
//    vector<vector<double>> X = b.get_X();
//    vector<int> y = b.get_y();
//    LogisticRegression lg2(X, y);
//    vector<int> results = LogisticRegression::predict(X);
//    cout << to_string(LogisticRegression::model_accuracy(results, b.get_y())) + "%";

}