# Курсовая работа 
## “Создание библиотеки для определение диабета при помощи обученной модели, основанной на логистической регрессии”


## Цель: 
Создать библиотеку, которая даст возможность обучить модель логистической регрессии на нахождение заболевания диабетом у человека.


## Библиотека состоит из 3-х классов: DiabetesData, LogisticRegression, Plot:


### Класс DiabetesData предоставляет возможность подготовить датасет для дальнейшей работы с классом LogisticRegression и Plot, путем разбиения датасета на двумерный массив double и одномерный массив int: массив X содержащий в себе строки датасета, не включая названия столбцов и исход(outcome), и у, содержащий в себе лишь исход(outcome).


### Устройство класса


#### Класс содержит в себе конструкторы:
```
DiabetesData(const string &data_name){
...
}
```
Этот конструктор принимает на вход название файла с датасетом, после чего он вызывает функцию ```load_data_from_file```, которая разбивает датасет на массивы X и y. В конце функции вызывается функция ```data_normalization```, которая нормализует матрицу X путем "z - масштабирования".
```
DiabetesData(const vector<vector<double>> &features){
...
} 
```
Этот конструктор принимает на вход двумерный масcив double состоящий из одной строчки и n столбцов (n = кол-ву столбцов датасета  - 1) и так же нормализует его с помощью ```data_normalization```.


### Класс LogisticRegression представляет собой реализацию модели, работающей на основе логистической регрессии.


### Устройство класса


#### Класс содержит в себе конструктор:
```
LogisticRegression(const vector<vector<double>> &X, const vector<int> &y)
```
Этот конструктор принимает на вход двумерный массив Х и одномерный массив у, сохраняет их и рандомит веса для будущей работы.


#### Класс содержит в себе функции:


logit - функция отвечает за подсчет логитов основываясь на датасете X и текущих весас и возвращает двумерный вектор double содержащий в себе логиты всех классов.


sigmoid - функция отвечает за подсчет сигмоид основываясь на логитах, возвращает двумерный вектор double содержащий в себе сигмоиды для каждого логита.


fit - функция, отвечающая за обучение нашей модели, путем корректировки весов с помощью градиентного спуска, который мы находим путем взятия производной от функции потерь. Возвращает вектор потерь типа double.


loss - функция, отвечающая за подсчет функции потерь нашей логистической регрессии с использованием l2 - регуляризации, основываясь на у исходе(outcome) и сигмоидах.


save_weights - функция отвечает за сохранение лучших весов найденных в fit().


predict_proba - функция отвечает за подсчет сигмоид с использованием наилучших найденных весов. Возвращает двумерный массив типа double, содержащий в себе сигмоды.


predict - функция отвечает за предсказание диагноза путем сравниния сигмоид(-ы) из функции predict_proba с 0.5, если больше, то диагноз положительный, если меньше отрицательный. Возвращает массив типа double, содержащий в себе результаты предсказаний.


model_accuracy - функция, показывающая точность модели с использованием f1 метрики. Возвращает целочисленное значение, процент (0; 70; 23...).


saveLossToCSV - функция, отвечающая за создание и сохранение потерь в csv файл, вида:
```
1,0 
0.80,1
0.5,2
...
n1, n2
```
где первый столбец — это столбец потерь, а второй — это кол-во итераций.


### Класс Plot предоставляет возможность создать «statistic.tex» файл, содержащий в себе небольшую, но информативную статистику модели, а именно график функции потерь и confusion matrix.


### Устройство класса


#### Класс содержит в себе 1 функцию:


CreateLatexFile — функция отвечает за создание "statistic.tex" файла.


![image](https://github.com/avanturer/DiabetesAI/assets/72664467/5d9b25ae-7fb2-4a28-80a5-e9f327d29330)
![image](https://github.com/avanturer/DiabetesAI/assets/72664467/c91c6739-9242-4111-a112-01f965712ebb)


### Оркин Р.Р
### ИТКН БПМ-22-4, НИТУ МИСиС
