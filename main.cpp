#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 
#include <math.h> 

using namespace std;

const int NUM_FEATURES = 48 * 48 + 1;
const int NUM_TRAIN_OBSERVATIONS = 3995 + 4097 + 1; //emotion 0 + emotion 2
const int NUM_TEST_OBSERVATIONS = 958 + 1054 + 1;
const int NUM_EPOCHS = 100;
const int COST_THRESHOLD = 0.001;
const double LEARNING_RATE = 0.1; // TODO: check value

double **allocMatrix(int rows, int cols){
    double **matrix = (double **)malloc(rows * sizeof(double *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (double *)malloc(cols * sizeof(double)); 

    return matrix;
}

void freeMatrix(double **matrix, int rows, int cols){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

void parsePixels(string pixels_str, double *pixels){
    stringstream ss(pixels_str);
    double pixel;
    int i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel / 255;
        i++;
    }
}

void initWeights(double *weights){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    uniform_real_distribution<> distribution(0.0, 1.0); // min = 0, max = 1

    for(int i = 0; i < NUM_FEATURES; i++){
        weights[i] = distribution(generator);
    }
}

double sigmoid(double z){
    return 1.0/(1.0+exp(-z));
}

double hipothesys(double *weights, double *observation){ //observation == xi
    double sumOfWeightedFeatures = 0, weight, feature;

    for (int i = 0; i < NUM_FEATURES; i++){
        weight = weights[i];
        feature = observation[i];

        sumOfWeightedFeatures += (weight * feature);
    }

    return sigmoid(sumOfWeightedFeatures);
}

void addToDataset(double **X, int *y, int index, int emotion, double* pixels, string usage){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[index][i] = pixels[i-1]; // pixels between 0 and 1
    }
}

double gradient(double **X_train, int *y_train, double *weights, int j){
    double h_xi = 0;
    double *xi;
    double sum = 0;
    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        xi = X_train[i];
        h_xi = hipothesys(weights, xi);
        sum += (h_xi - y_train[i])*xi[j];
    }
    return (-1/NUM_TRAIN_OBSERVATIONS) * sum;
}

void updateWeights(double **X_train, int *y_train, double *weights, double *newWeights){
    
    for(int i = 0; i < NUM_FEATURES; i++){
	    newWeights[i] = weights[i] - (LEARNING_RATE * gradient(X_train, y_train, weights, i));
    }

    for(int i = 0; i < NUM_FEATURES; i++){
        weights[i] = newWeights[i];
    }
}

double cost_function(double **X_train, int *y_train, double *weights){
    double cost = 0;
    double h_xi;

    for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        h_xi = hipothesys(weights, X_train[i]);
        if(y_train[i] == 1){
            cost +=  y_train[i] * log(h_xi);
        }else{
            cost +=  (1-y_train[i]) * log(1-h_xi);
        }
        //cost += ((-y_train[i]*log(h_xi)) - ((1-y_train[i])*log(1-h_xi)));
    }

    return (-1/NUM_TRAIN_OBSERVATIONS) * cost;
}


int main(){
    double **X_train, **X_test, *weights, *newWeights;
    int *y_train, *y_test;
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    y_train = (int *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(int));
    y_test = (int *)malloc(NUM_TEST_OBSERVATIONS * sizeof(int));

    weights = (double *) malloc (NUM_FEATURES * sizeof(double));
    initWeights(weights);

    newWeights = (double *) malloc (NUM_FEATURES * sizeof(double));

    ifstream inputFile;
    inputFile.open("data/images.csv");
    string line;
    int index_train = 0, index_test = 0;
    double pixels[48*48];
    int a;
    getline(inputFile, line); // skip first line
    
    // Reading input
    while(getline(inputFile, line)){
        int lastPixelIndex = line.find(",", 2);
        int emotion = line[0] - '0';
        string pixels_str = line.substr(2, line.find(",", 2)-2);
        string usage = line.substr(lastPixelIndex+1);
        parsePixels(pixels_str, pixels);

        if(emotion == 0 || emotion == 2){
                emotion = round(emotion / 2);
            if(usage.compare("Training") != 0){
                addToDataset(X_test, y_test, index_test, emotion, pixels, usage);
                index_test ++;
            }else{
                addToDataset(X_train, y_train, index_train, emotion, pixels, usage);
                index_train ++;
            }
        }
    }

    inputFile.close();
    double cost = INT_MAX;
    double newCost = cost_function(X_train, y_train, weights);
    int epoch = 0;
    
    while(epoch < NUM_EPOCHS){
        double predictions[NUM_TEST_OBSERVATIONS], pred;
        int correct = 0;
        for (int i = 0; i < NUM_TEST_OBSERVATIONS; i++){
            pred = round(hipothesys(weights, X_test[i]));
            predictions[i] = pred;
            if(pred == y_test[i]){
                correct ++;
            }
        }

        cout<<"correct: "<<correct<<" wrong: "<<NUM_TEST_OBSERVATIONS-correct << " " <<((float)correct/NUM_TEST_OBSERVATIONS)*100<<"%"<<endl;



        cout << "Processando época: " << epoch << ", Custo: " <<cost<< endl;
        updateWeights(X_train, y_train, weights, newWeights);
        cost = newCost;
        newCost = cost_function(X_train, y_train, weights);
        epoch ++;
    }

    

    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    free(y_train);
    free(y_test);
}
