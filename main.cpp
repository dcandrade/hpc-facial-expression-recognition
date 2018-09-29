#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 

using namespace std;

const int NUM_FEATURES = 48 * 48 + 1;
const int NUM_TRAIN_OBSERVATIONS = 28709 + 1;
const int NUM_TEST_OBSERVATIONS = 3589*2 + 1;

int **allocMatrix(int rows, int cols){
    int **matrix = (int **)malloc(rows * sizeof(int *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (int *)malloc(cols * sizeof(int)); 

    return matrix;
}

void freeMatrix(int** matrix, int rows, int cols){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

void parsePixels(string pixels_str, int *pixels){
    stringstream ss(pixels_str);
    int pixel, i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel;
        i++;
    }
}

void addToDataset(int **X, int *y, int index, int emotion, int* pixels, string usage){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i <= NUM_FEATURES; i++){
        X[index][i] = pixels[i-1]; //TODO: normalize pixel valus
    }
}

int main(){
    int **X_train, **X_test, *y_train, *y_test;
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    y_train = (int *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(int));
    y_test = (int *)malloc(NUM_TEST_OBSERVATIONS * sizeof(int));

    ifstream inputFile;
    inputFile.open("data/images.csv");
    string line;
    int index_train = 0, index_test = 0;
    int pixels[48*48];
    int a;
    getline(inputFile, line); // skip first line
    
    while(getline(inputFile, line)){
        int lastPixelIndex = line.find(",", 2);
        int emotion = line[0] - '0';
        string pixels_str = line.substr(2, line.find(",", 2)-2);
        string usage = line.substr(lastPixelIndex+1);
        parsePixels(pixels_str, pixels);

        if(usage.compare("Training") != 0){
            addToDataset(X_test, y_test, index_test, emotion, pixels, usage);
            index_test ++;
        }else{
            addToDataset(X_train, y_train, index_train, emotion, pixels, usage);
            index_train ++;
        }

    }

    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    free(y_train);
    free(y_test);
}
