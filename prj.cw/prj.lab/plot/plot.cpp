//
// Created by hedge on 13.05.2023.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <plot/plot.hpp>

void Plot::CreateLatexFile(vector<int> results, vector<int> y, int max_iter) {
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
