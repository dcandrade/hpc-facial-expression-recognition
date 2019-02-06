/**
 * @file main.cpp
 * @author Daniel Andrade e Gabriel Gomes
 * @brief Reconhecedor de Expressões Faciais através de Regressão Logística
 * @version 1.0
 * @date 2018-10-12
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 
#include <math.h> 
#include <time.h>
#include <CL/cl.h>


const char *KernelSource = "\n" \
"__kernel void vadd(                                                    \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";


#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif


using namespace std;

int NUM_FEATURES = 128 * 128 + 1; /// Quantidade de pixels + bias
int NUM_TRAIN_OBSERVATIONS = 15640; /// Quantidade de observações de treino
int NUM_TEST_OBSERVATIONS = 3730; /// Quantidade de observações de teste
int NUM_EPOCHS = 500; /// Quantidade de épocas
int NUM_ITERATIONS = 5;
float LEARNING_RATE = 0.001;	 /// Taxa de Apredizado
string OUTPUT_FILE_PREFIX = "results/"; /// Prefixo do arquivo de saída

/**
 * @brief Aloca uma dinamicamente uma matriz de floats e retorna o ponteiro para acessá-la
 * 
 * @param rows Quantidade de linhas 
 * @param cols Quantidade de colunas
 * @return float** 
 */
float **allocMatrix(int rows, int cols){
    float **matrix = (float **)malloc(rows * sizeof(float *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (float *)malloc(cols * sizeof(float)); 

    return matrix;
}

/**
 * @brief Libera uma matriz dinamicamente alocada
 * 
 * @param matrix Ponteiro para a matriz a ser liberada
 * @param rows Quantidade de linhas da matriz
 */
void freeMatrix(float **matrix, int rows){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

/**
 * @brief Popula e normaliza um vetor de floats com os pixels presentes numa string
 * 
 * @param pixels_str String com os pixels separados por espaço
 * @param pixels Vetor de 48*48 espaços nos quais os pixels serão armazenados
 */
void parsePixels(string pixels_str, float *pixels){
    stringstream ss(pixels_str);
    float pixel;
    int i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel / 255;
        i++;
    }
}

/**
 * @brief 
 * 
 * @param X Matriz de características
 * @param y Vetor de marcações (gabaritos)
 * @param index Índice na observação/marcação atual
 * @param sex Emoção normalizada (0 ou 1)
 * @param pixels Vetor de 48*48 posições contendo os valores normalizado de cada pixel
 */
void addToDataset(float **X, float *y, int index, int sex, float* pixels){
    X[index][0] = 1;
    y[index] = sex;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[index][i] = pixels[i-1];
    }
}

/**
 * @brief Calcula função logística ou função sigmóide
 * 
 * @param z Parâmetro
 * @return float Valor da função para z
 */
float sigmoid(float z){
    return 1.0/(1.0+exp(-z));
}

/**
 * @brief Calcula o ŷ para uma determinada observação dado um conjunto de pesos
 * 
 * @param weights Pesos a serem aplicados na hipótese
 * @param observation Observação para qual o ŷ será calculado
 * @return float ŷ  
 */
float hypothesis(float *weights, float *observation){ //observation == xi
    float z = 0;

    for (int i = 0; i < NUM_FEATURES; i++){
        z += (weights[i] * observation[i]);								/// Este produto representa a hipótese
    }

    return sigmoid(z);													///A hipótese é passada para função sigmóide antes de retornar, obtendo ŷ
}
//Método usado para calcular a função de custo.
float cost_function(float **X_train, float *y_train, float *predictions){
    float cost = 0;
    float h_xi;

    for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        h_xi = predictions[i];
        float p1 = y_train[i] * log(h_xi);								///primeira parte da função de custo
        float p2 = (1-y_train[i]) * log(1-h_xi);						///segunda parte da função de custo
        cost += (-p1-p2);												///função de custo dada pelo somatório do inverso das duas partes
    }

    return  cost/NUM_TRAIN_OBSERVATIONS;
}
//Método para calcular o gradiente. É passado como parâmetro todas as observações, as respostas e as predições.
float gradient(float **X_train, float *y_train, float *predictions, int j){
    float h_xi = 0;
    float *xi;
    float sum = 0;
    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        xi = X_train[i];										///pega uma das observações
        h_xi = predictions[i];									///pega uma das predições (resposta prevista)
        sum += (h_xi - y_train[i])*xi[j];						///faz o somatório da diferença do vetor de resposta prevista pelo vetor de resposta multiplicado pelo vetor contendo os valores de observação
    }
    return (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * sum;
}

/**
 * @brief Recalcula e atualiza os coeficientes de regressão (pesos)
 * 
 * @param X_train Conjunto de observações
 * @param y_train Marcações das observações
 * @param weights Pesos atuais
 * @param predictions Predicões de cada item de X_train utilizando os coeficientes mais recentes
 */
void updateWeights(float **X_train, float *y_train, float *weights, float *predictions){    
    for(int j = 0; j < NUM_FEATURES; j++){
	    weights[j] -= gradient(X_train, y_train, predictions, j);
    }
}

/**
 * @brief Salva as informações de uma época no arquivo de saída
 * 
 * @param epoch Número da época atual
 * @param outputFile Arquivo de saída
 * @param predictions Predições atuais
 * @param y Marcação das observações
 * @param size Tamanho do vetor y
 * @param cost Custo do do erro da época atual
 */
void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost){
    //cout << "done epoch #" <<epoch<<endl;
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for(int i = 0; i < size; i++){										///calculando dados para matriz de confusão a partir do valor real e do valor predito
        pred = round(predictions[i]);
        real= round(y[i]);

        if(pred == 0 && real == 0)
                tn++;
        else if(pred == 0 && real == 1)
                fn++;
        else if(pred == 1 && real == 0)
                fp++;
        else
                tp++;
    }
    
    accuracy = (tp + tn)/(tp + fp + fn + tn);
    precision = tp/(tp + fp);
    recall = tp/(tp + fn);
    f1 = (2*recall*precision)/(recall + precision);

    outputFile << epoch <<  "," << accuracy <<  "," << precision << "," << recall <<  "," << f1 <<  "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv){
    if(argc < 5){
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA"<<endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]);
    NUM_TEST_OBSERVATIONS = atof(argv[2]);
    NUM_EPOCHS = atof(argv[3]);
    LEARNING_RATE = atof(argv[4]);
    OUTPUT_FILE_PREFIX += argv[5];
    OUTPUT_FILE_PREFIX += "/";
}

int main(int argc, char** argv){
    parse_args(argc, argv);
    int error;

    /******** Initial Variables ********/
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceIdCount);

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

    int i =0;
    cl_char string[10240] = {0};
    // Print out the platform name
    error = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
    //checkError(error, "Getting platform name");
    printf("Platform: %s\n", string);

    // Print out the platform vendor
    error = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
    //checkError(error, "Getting platform vendor");
    printf("Vendor: %s\n", string);

    // Print out the platform OpenCL version
    error = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
    //checkError(err, "Getting platform OpenCL version");
    printf("Version: %s\n", string);


    const cl_context_properties contextProperties [] = {CL_CONTEXT_PLATFORM,reinterpret_cast<cl_context_properties> (platformIds [0]), 0, 0};


    cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);

    if(error != CL_SUCCESS){
        cout << "OpenCL error while creating context " << error << endl;
        exit(EXIT_FAILURE);
    }else{
        cout << "OpenCL context created" << endl;
    }

    /******** End Initial Variables ********/


    /******** Buffers *******
    cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float), nullptr, &error);
    
    cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TRAIN_OBSERVATIONS * sizeof(float), nullptr, &error);

    cl_mem buffer_X_test = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float), nullptr, &error);
    
    cl_mem buffer_y_test = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TEST_OBSERVATIONS * sizeof(float), nullptr, &error);
    /******** End Buffers ********/


    if(error != CL_SUCCESS){
        cout << "OpenCL error while allocating buffers" << error << endl;
        exit(EXIT_FAILURE);
    }else{
        cout << "OpenCL buffers done" << endl;
    }

        cl_uint num_devices;
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

        // Get the device IDs
        cl_device_id device[num_devices];
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
        //checkError(err, "Getting devices");
        printf("Number of devices: %d\n", num_devices);

        // Investigate each device
        for (int j = 0; j < num_devices; j++)
        {
            printf("\t-------------------------\n");

            // Get device name
            error = clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
            //checkError(err, "Getting device name");
            printf("\t\tName: %s\n", string);

            // Get device OpenCL version
            error = clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
            //checkError(err, "Getting device OpenCL C version");
            printf("\t\tVersion: %s\n", string);

            // Get Max. Compute units
            cl_uint num;
            error = clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
            //checkError(err, "Getting device max compute units");
            printf("\t\tMax. Compute Units: %d\n", num);

            // Get local memory size
            cl_ulong mem_size;
            error = clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            //checkError(err, "Getting device local memory size");
            printf("\t\tLocal Memory Size: %llu KB\n", mem_size/1024);

            // Get global memory size
            error = clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            //checkError(err, "Getting device global memory size");
            printf("\t\tGlobal Memory Size: %llu MB\n", mem_size/(1024*1024));

            // Get maximum buffer alloc. size
            error = clGetDeviceInfo(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            //checkError(err, "Getting device max allocation size");
            printf("\t\tMax Alloc Size: %llu MB\n", mem_size/(1024*1024));

            // Get work-group size information
            size_t size;
            error = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
            //checkError(err, "Getting device max work-group size");
            printf("\t\tMax Work-group Total Size: %ld\n", size);

            // Find the maximum dimensions of the work-groups
            error = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
            //checkError(err, "Getting device max work-item dims");

            // Get the max. dimensions of the work-groups
            size_t dims[num];
            error = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
            //checkError(err, "Getting device max work-item sizes");
            printf("\t\tMax Work-group Dims: ( ");
            for (size_t k = 0; k < num; k++)
            {
                printf("%ld ", dims[k]);
            }
            printf(")\n");

            error = clGetDeviceInfo(device[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(size_t), &size, NULL);

            printf("\t\tPreferred vector width (float): %ld\n", size);


            printf("\t-------------------------\n");

        

        }
}
