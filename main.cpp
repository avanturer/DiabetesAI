//
// Created by hedge on 22.04.2023.
//

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

    explicit DiabetesData(const string &data_name) {
        load_data_from_file(data_name);
    }

    void load_data_from_file(const string &data_name) {
        fstream fin;
        fin.open(data_name + ".csv", ios::in);
        string line;
        vector<vector<string>> parsedCsv;
        if (fin.fail()) {
            cout << "NOT OPEN";
        }
        while (getline(fin, line)) {
            stringstream lineStream(line);
            string cell;
            vector<string> parsedRow;
            while (getline(lineStream, cell, ',')) {
                parsedRow.push_back(cell);
            }

            parsedCsv.push_back(parsedRow);
        }
        fin.close();
        dataset_ = parsedCsv;

        for (int i = 1; i < dataset_.size(); i++) {
            vector<double> cell;
            for (int j = 0; j < dataset_[1].size(); j++) {
                if (j != dataset_[1].size() - 1) {
                    cell.push_back(stod(dataset_[i][j]));
                } else if (j == dataset_[1].size() - 1) {
                    y_.push_back(stoi(dataset_[i][j]));
                }
            }
            X_.push_back(cell);
        }
        X_ = data_normalization(X_);
    }

    explicit DiabetesData(const vector<vector<double>> &features) {
        if (features.size() != dataset_[0].size() || features.empty())
            throw runtime_error(
                    "The number of parameters does not match the number of dataset parameters, check the correctness of the entered data and check which dataset you submitted");

        features_ = data_normalization(features);
    }


    static vector<vector<double>> data_normalization(vector<vector<double>> X) {

        vector<double> avarage;
        for (int j = 0; j < X[0].size(); j++) {
            double summ = 0;
            for (auto &i: X) {
                summ += i[j];
            }
            avarage.push_back(summ / X.size());
        }

        vector<double> deviation;
        for (int j = 0; j < X[0].size(); j++) {
            double summ = 0;
            for (auto &i: X)
                summ += pow((i[j] - avarage[j]), 2);

            deviation.push_back(sqrt(summ / (X.size() - 1)));
        }

        for (int j = 0; j < X[0].size(); j++) {
            for (auto &i: X) {
                i[j] = (i[j] - avarage[j]) / deviation[j];
                if (i[j] > 3 * deviation[j]) {
                    i[j] = 0;
                }
            }
        }
        return X;
    }

    vector<vector<double>> features_;
    vector<vector<string>> dataset_;
    vector<vector<double>> X_;
    vector<int> y_;
private:

};

class LogisticRegression {
public:
    vector<vector<double>> X_;
    vector<int> y_;
    vector<double> w_;
    vector<double> losses_;
    vector<vector<double>> logloss_;
    int max_iter_ = 0;
    double lr_ = 0;


    LogisticRegression(const vector<vector<double>> &X, const vector<int> &y) {
        X_ = X;
        y_ = y;
        for (int i = 0; i < X[0].size() + 1; i++) {
            double a = rand() % 1000;
            w_.push_back(a / 1000);
        }

    }

    static vector<vector<double>> logit(vector<vector<double>> X, vector<double> w) {
        unsigned long long int m = X.size();
        unsigned long long int n = X[0].size();
        vector<vector<double>> logits(m, vector<double>(1, 0.0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < n; k++) {
                    logits[i][j] += X[i][k] * w[k];
                }
            }
        }
        return logits;
    }

    static vector<vector<double>> sigmoid(vector<vector<double>> logits) {
        vector<vector<double>> sigmoids;
        for (int i = 0; i < logits.size(); i++) {
            vector<double> sigmoid;
            sigmoid.reserve(logits[0].size());
            for (int j = 0; j < logits[0].size(); j++)
                sigmoid.push_back(1 / (1 + exp(-logits[i][j])));

            sigmoids.push_back(sigmoid);
        }
        return sigmoids;
    }

    vector<double> fit(int max_iter = 100, double lr = 0.1) {
        if (max_iter <= 0 || lr <= 0) {
            throw runtime_error("maximum iterations or learning rate < 0");
        }
        max_iter_ = max_iter;
        lr_ = lr;
        vector<vector<double>> X_train = X_;
        for (auto &i: X_train)
            i.insert(i.begin(), 1);


        vector<vector<double>> X_trainT(X_train[0].size(), vector<double>(X_train.size(), 0.0));
        for (int i = 0; i < X_train.size(); i++) {
            for (int j = 0; j < X_train[0].size(); j++) {
                X_trainT[j][i] = X_train[i][j];
            }
        }
        for (int iter = 0; iter <= max_iter_; iter++) {
            vector<vector<double>> z;

            for (int i = 0; i < sigmoid(logit(X_train, w_)).size(); i++) {
                z.push_back(sigmoid(logit(X_train, w_))[i]);
            }
            unsigned long long int m = X_trainT.size();
            unsigned long long int n = X_trainT[0].size();
            unsigned long long int p = z[0].size();

            vector<vector<double>> grad(m, vector<double>(p, 0.0));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    for (int k = 0; k < n; k++) {
                        grad[i][j] += ((X_trainT[i][k] * (z[k][0] - y_[k])) + 2 * w_[i]);
                    }
                    grad[i][j] /= y_.size();
                }
            }

            for (int i = 0; i < w_.size(); i++) {
                w_[i] -= (grad[i][0] * lr_);
            }
            losses_.push_back(loss(y_, z));
            if ((!losses_.empty()) && (loss(y_, z) < *min_element(losses_.begin(), losses_.end()))) {
                save_weights(w_);
            }
        }
        saveLossToCSV(losses_);
        return losses_;
    }

    double loss(vector<int> y, vector<vector<double>> z) {
        double loss = 0;
        for (int i = 0; i < y.size(); i++) {
            loss += (y[i] * (log(z[i][0])) + (1 - y[i]) * log(1 - z[i][0]));
        }
        for (double i: w_)
            loss += pow(i, 2);

        loss /= y.size();
        loss *= -1;
        return loss;
    }

    static void save_weights(const vector<double> &weights) {
        ofstream outfile("weights.txt", ios::out | ios::trunc);
        bool file_exists = outfile.good();

        if (!file_exists) {
            ofstream outfile("weights.txt");
            outfile.close();
        }

        ofstream file("weights.txt", ios_base::app);

        for (double w: weights) {
            file << w << endl;
        }
        file.close();
    }

    static vector<vector<double>> predict_proba(vector<vector<double>> feauters) {
        vector<vector<double>> _f = move(feauters);
        vector<double> w;
        for (auto &i: _f)
            i.insert(i.begin(), 1);

        fstream fin;
        fin.open("weights.txt", ios::in);
        string line;
        while (getline(fin, line)) {
            stringstream lineStream(line);
            string cell;
            getline(lineStream, cell);
            w.push_back(stod(cell));

        }
        fin.close();
        return sigmoid(logit(_f, w));
    }

    static vector<int> predict(const vector<vector<double>> &feauters, double threshold = 0.5) {
        vector<int> results;
        for (int i = 0; i < predict_proba(feauters).size(); i++) {
            results.push_back(predict_proba(feauters)[i][0] >= threshold);
        }
        return results;
    }

    static int model_accuracy(vector<int> results, vector<int> y) {
        double accuracy = 0;
        double YES = 0;
        double NO = 0;
        for (int i = 0; i < results.size(); i++) {
            if (results[i] == y[i]) {
                YES++;
            } else {
                NO++;
            }
        }
        accuracy = YES / (YES + NO);
        return round(accuracy * 100);
    }

    static void saveLossToCSV(const vector<double> &losses) {
        const string filename = "losses_.csv";
        ofstream outputFile(filename);
        if (outputFile.is_open()) {
            outputFile << 1 << "," << 0 << endl;
            for (int i = 0; i < losses.size(); i++) {
                outputFile << losses[i] << "," << i + 1 << endl;
            }
            outputFile.close();
            cout << "Data saved to losses_.csv successfully." << endl;
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

};

class Plot {
public:
    Plot() = default;

    static void CreateLatexFile(vector<int> results, vector<int> y, int max_iter = 100) {
        int YESaYES = 0;
        int YESaNO = 0;
        int NOaNO = 0;
        int NOaYES = 0;
        for (int i = 0; i < results.size(); i++) {
            if (results[i] == 1 && y[i] == 1) {
                YESaYES++;
            } else if (results[i] == 1 && y[i] == 0) {
                YESaNO++;
            } else if (results[i] == 0 && y[i] == 0) {
                NOaNO++;
            } else if (results[i] == 0 && y[i] == 1) {
                NOaYES++;
            }
        }
        const string &filename = "statistic.tex";
        ofstream outputFile(filename);
        if (outputFile.is_open()) {
            outputFile << "\\documentclass{article}\n"
                          "\\usepackage{pgfplots}\n"
                          "\\usepackage{csvsimple}\n"
                          "\\usepackage{pgfplotstable}\n"
                          "\\usepackage{array}\n"
                          "\\usepackage{graphicx}\n"
                          "\\usepackage{multirow}\n"
                          "\\usepackage{float} % Add the float package\n"
                          "\n"
                          "\\newcommand\\MyBox[2]{\n"
                          "  \\fbox{\\lower0.75cm\n"
                          "    \\vbox to 1.7cm{\\vfil\n"
                          "      \\hbox to 1.7cm{\\hfil\\parbox{0.4cm}{#1}\\hfil}\n"
                          "      \\vfil}%\n"
                          "  }%\n"
                          "}\n"
                          "\n"
                          "\\begin{document}\n"
                          "\\title{Model Statistic}\n"
                          "\\maketitle\n"
                          "\n"
                          "\\vspace{1cm}\n"
                          "\n"
                          "\\begin{figure}[H] % Use the H specifier from the float package\n"
                          "  \\centering\n"
                          "  \\begin{tikzpicture}\n"
                          "    \\begin{axis}[\n"
                          "        xlabel={Iters},\n"
                          "        ylabel={Log Loss},\n"
                          "        xtick=data,\n"
                          "        xticklabel={\\pgfmathprintnumber[int detect]{\\tick}},"
                          "        xmin=0, xmax=" + to_string(max_iter) + ",\n"
                                                                          "        ymin=0, ymax=1,\n"
                                                                          "        grid=both,\n"
                                                                          "        width=14cm, height=8cm,\n"
                                                                          "        samples=100\n"
                                                                          "    ]\n"
                                                                          "    \n"
                                                                          "    \\pgfplotstableread[col sep=comma]{losses_.csv}\\datatable\n"
                                                                          "    \\addplot[blue] table[x index=1, y index=0] {\\datatable};\n"
                                                                          "    \n"
                                                                          "    \\end{axis}\n"
                                                                          "  \\end{tikzpicture}\n"
                                                                          "  \\caption{Loss changes every iteration}\n"
                                                                          "\\end{figure}\n"
                                                                          "\n"
                                                                          "\\vspace{1cm}\n"
                                                                          "\n"
                                                                          "\\begin{figure}[H] % Use the H specifier from the float package\n"
                                                                          "  \\centering\n"
                                                                          "  \\renewcommand\\arraystretch{1.5}\n"
                                                                          "  \\setlength\\tabcolsep{0pt}\n"
                                                                          "  \\begin{tabular}{c >{\\bfseries}r @{\\hspace{0.7em}}c @{\\hspace{0.4em}}c @{\\hspace{0.7em}}l}\n"
                                                                          "    \\multirow{10}{*}{\\rotatebox{90}{\\parbox{1.1cm}{\\bfseries\\centering actual\\\\ value}}} & \n"
                                                                          "      & \\multicolumn{2}{c}{\\bfseries Prediction outcome} & \\\\\n"
                                                                          "    & & \\bfseries p & \\bfseries n & \\bfseries total \\\\\n"
                                                                          "    & p$'$ & \\MyBox{" + to_string(YESaYES) +
                          "}{Positive} & \\MyBox{" + to_string(NOaYES) +
                          "}{Positive} & P$'$ \\\\[2.4em]\n"
                          "    & n$'$ & \\MyBox{" + to_string(YESaNO) + "}{Positive} & \\MyBox{" + to_string(NOaNO) +
                          "}{Positive} & N$'$ \\\\\n"
                          "    & total & P & N &\n"
                          "  \\end{tabular}\n"
                          "  \\caption{Confusion Matrix}\n"
                          "\\end{figure}\n"
                          "\n"
                          "\\end{document}";
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

private:
};

int main() {
    DiabetesData a("dataset1");
    vector<vector<double>> X1 = a.X_;
    vector<int> y1 = a.y_;
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit(5);
    for (double losse: losses)
        cout << losse << endl;

    DiabetesData b("dataset2");
    vector<vector<double>> X = b.X_;
    vector<int> y = b.y_;
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
    Plot::CreateLatexFile(results, y, 5);


    system("pdflatex statistic.tex");


}