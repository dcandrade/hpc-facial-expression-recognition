/**
 * @file main.cpp
 * @author Daniel Andrade e Gabriel Gomes
 * @brief Reconhecedor de Expressões Faciais através de Regressão Logística - Versão GPU
 * @version 1.0
 * @date 2019-05-02
 * 
 * @copyright Copyright (c) 2019
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

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

using namespace std;

int NUM_FEATURES = 128 * 128 + 1;   /// Quantidade de pixels + bias
int NUM_TRAIN_OBSERVATIONS = 15640; /// Quantidade de observações de treino
int NUM_TEST_OBSERVATIONS = 3730;   /// Quantidade de observações de teste
int NUM_EPOCHS = 500;               /// Quantidade de épocas
int NUM_ITERATIONS = 5;             /// Quantidade de vezes que o experimento será repetido
float LEARNING_RATE = 0.001;            /// Taxa de Apredizado
string OUTPUT_FILE_PREFIX = "results/"; /// Prefixo do arquivo de saída
int LOCAL_WORK_SIZE = 256;

/**
 * @brief Carrega um kernel OpenCL em um buffer
 * 
 * @param path Caminho do arquivo contendo o kernel OpenCL
 * @param Buffer onde o kernel será armazenado
 * @source  http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
 **/
long LoadOpenCLKernel(char const *path, char **buf){
    FILE *fp;
    size_t fsz;
    long off_end;
    int rc;

    /* Open the file */
    fp = fopen(path, "r");
    if (NULL == fp)
    {
        return -1L;
    }
    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if (0 != rc)
    {
        return -1L;
    }
    /* Byte offset to the end of the file (size) */
    if (0 > (off_end = ftell(fp)))
    {
        return -1L;
    }
    fsz = (size_t)off_end;
    /* Allocate a buffer to hold the whole file */
    *buf = (char *)malloc(fsz + 1);
    if (NULL == *buf)
    {
        return -1L;
    }
    /* Rewind file pointer to start of file */
    rewind(fp);
    /* Slurp file into buffer */
    if (fsz != fread(*buf, 1, fsz, fp))
    {
        free(*buf);
        return -1L;
    }
    /* Close the file */
    if (EOF == fclose(fp))
    {
        free(*buf);
        return -1L;
    }
    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';
    /* Return the file size */
    return (long)fsz;
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
    int i = 0;

    while (ss >> pixel)
    {
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
void addToDataset(float *X, float *y, int index, int sex, float* pixels){
    int maskedIndex = index * NUM_FEATURES;
    X[maskedIndex] = 1;
    y[index] = sex;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[maskedIndex + i] = pixels[i-1];
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
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for (int i = 0; i < size; i++){ ///calculando dados para matriz de confusão a partir do valor real e do valor predito
        pred = round(predictions[i]);
        real = round(y[i]);

        if (pred == 0 && real == 0)
            tn++;
        else if (pred == 0 && real == 1)
            fn++;
        else if (pred == 1 && real == 0)
            fp++;
        else
            tp++;
    }

    // Cálculo das métricas de avaliação
    accuracy = (tp + tn) / (tp + fp + fn + tn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = (2 * recall * precision) / (recall + precision);

    cout << epoch << "," << accuracy << "," << precision << "," << recall << "," << f1 << "," << cost << endl;
    outputFile << epoch << "," << accuracy << "," << precision << "," << recall << "," << f1 << "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv){
    if (argc < 7){
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA" << endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]);
    NUM_TEST_OBSERVATIONS = atof(argv[2]);
    NUM_EPOCHS = atof(argv[3]);
    LEARNING_RATE = atof(argv[4]);
    OUTPUT_FILE_PREFIX += argv[5];
    OUTPUT_FILE_PREFIX += "/";
    LOCAL_WORK_SIZE = atof(argv[6]);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    
    /// Alocação das estruturas para armazenamento do dataset
    float *X_train, *X_test;						
    float *y_train, *y_test;	

    X_train = (float *) malloc(NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float));
    X_test = (float *) malloc(NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float));    			
    y_train = (float *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));		 
    y_test = (float *)malloc(NUM_TEST_OBSERVATIONS * sizeof(float));		
    
    ifstream trainFile, testFile;
    trainFile.open("../data/train.csv");
    testFile.open("../data/test.csv");

    if(!trainFile.is_open()){
        cout <<" O arquivo de entrada data/train.csv não foi encontrado"<<endl;
        exit(EXIT_FAILURE);
    }

    if(!testFile.is_open()){
        cout <<" O arquivo de entrada data/test.csv não foi encontrado"<<endl;
        exit(EXIT_FAILURE);
    }

    string line;
    int index = 0, sex, pixelsIndex;
    float pixels[NUM_FEATURES - 1];

    while(getline(trainFile, line)){    /// Carregamento do arquivo de entrada (treino)
        sex = line[0] - '0';
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TRAIN_OBSERVATIONS){
            addToDataset(X_train, y_train, index, sex, pixels);
            index ++;
        }
    }

    index = 0;

    while(getline(testFile, line)){    /// Carregamento do arquivo de entrada (teste)
        sex = line[0] - '0';
        
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TEST_OBSERVATIONS){
            addToDataset(X_test, y_test, index, sex, pixels);
            index ++;
        }
    }

    trainFile.close();
    testFile.close(); 
    
    /******** Variáveis iniciais do OpenCL ********/
    size_t localWorkSize = LOCAL_WORK_SIZE;
    size_t globalWorkSize = LOCAL_WORK_SIZE;

    int err;

    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);

    cl_platform_id platform_ids[100];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;          

    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); /// Realização da conexão com a GPU
    if (err != CL_SUCCESS){
        cout << "Error: Failed to create a device group!" << endl;
        exit(EXIT_FAILURE);
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err); /// Criação do contexto
    if (!context){
        cout << "Error: Failed to create a compute context!" << endl;
        exit(EXIT_FAILURE);
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err); /// Criação da fila de comandos
    if (!commands){
        cout << "Error: Failed to create a command commands!" << endl;
        exit(EXIT_FAILURE);
    }

    /// Criação do programa através do arquivo que descreve o kernel que será utilizado
    char *KernelSource;
    long lFileSize;

    lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
    if( lFileSize < 0L ) {
         cout << "File read failed" << endl;
        exit(EXIT_FAILURE);
    }

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program){
         cout << "Error: Failed to create compute program!" << endl;
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        cout << "Error: Failed to build program executable!" << endl;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        cout <<  buffer << endl;
        exit(EXIT_FAILURE);
    }


    kernel = clCreateKernel(program, "logreg", &err); /// Criação do kernel que será executado no device
    if (!kernel || err != CL_SUCCESS){
        cout << "Error: Failed to create compute kernel!" << endl;
        exit(EXIT_FAILURE);
    }

    /// Configuração dos buffers
    int X_size = NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float); /// Tamanho do buffer das amostras de treino
    int y_size = NUM_TRAIN_OBSERVATIONS * sizeof(float); // Tamanho do buffer das marcações de treino

    int X_size_test = NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float); /// Tamanho do buffer das amostras de teste
    int y_size_test = NUM_TEST_OBSERVATIONS * sizeof(float); // Tamanho do buffer das marcações de teste

    cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, X_size, X_train, &err); /// Buffer das amostras de treino

    if(err){
        cout << "Error: Failed to allocate device memory! (X_train, " << err <<")" << endl;
    }

    cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, y_size, y_train, &err); /// Buffer das marcações de treino

    if(err){
        cout << "Error: Failed to allocate device memory! (y_train, " << err <<")" << endl;
    }

    cl_mem buffer_X_test = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, X_size_test, X_test, &err); /// Buffer das amostras de teste

    if(err){
        cout << "Error: Failed to allocate device memory! (X_train, "<<err <<")" << endl;
    }

    cl_mem buffer_y_test = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, y_size_test, y_test, &err); /// Buffer das marcações de teste

    if(err){
        cout << "Error: Failed to allocate device memory! (y_train, "<<err <<")" << endl;
    }

    cl_mem buffer_weigths =  clCreateBuffer(context,  CL_MEM_READ_WRITE, NUM_FEATURES * sizeof(float), NULL, &err); /// Buffer dos pesos do algoritmo de regressão logística

    if(err){
        cout << "Error: Failed to allocate device memory! (weigths, "<<err <<")" << endl;
    }


    cl_mem buffer_predictions =  clCreateBuffer(context, CL_MEM_READ_WRITE, y_size, NULL, &err); /// Buffer para armazenar as predições realizadas pelo algoritmo

    if(err){
        cout << "Error: Failed to allocate device memory! (preds, "<<err <<")" << endl;
    }

    if(!buffer_X_train                 || 
       !buffer_y_train                 ||
       !buffer_X_test                  || 
       !buffer_y_test                  ||
       !buffer_weigths                 ||
       !buffer_predictions             ||
       err != CL_SUCCESS){
        
        cout << "Error: Failed to allocate device memory!" << endl;
        exit(EXIT_FAILURE);
    }   

    float *predictions = (float*) malloc(y_size);
    /// Argumentos padrão do kernel
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &buffer_weigths);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &buffer_predictions);
    err |= clSetKernelArg(kernel, 4, localWorkSize * NUM_FEATURES * sizeof(float), NULL);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *) &NUM_FEATURES);
    err |= clSetKernelArg(kernel, 8, sizeof(float), (void *) &NUM_EPOCHS);

    if (err != CL_SUCCESS){
        cout << "Error: Failed to set kernel arguments! " << err << endl;
        exit(EXIT_FAILURE);
    }

    /// Repete os experimentos por NUM_ITERATIONS vezes
    for(int iterations= 0; iterations< NUM_ITERATIONS; iterations++){
        cout << "Iteração "<< iterations << endl;
        // Configuração dos argumentos para a realização do treino
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buffer_X_train);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &buffer_y_train);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *) &NUM_TRAIN_OBSERVATIONS);
        err |= clSetKernelArg(kernel, 7, sizeof(float), (void *) &LEARNING_RATE);

       
        if (err != CL_SUCCESS){
            cout << "Error: Failed to set kernel arguments! " << err << endl;
            exit(EXIT_FAILURE);
        }

        auto start = chrono::system_clock::now();

        /// Comando para a execução do treinamento
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
       
        if (err != CL_SUCCESS){
            cout << "Error: Failed to execute kernel (train)! " << err << endl;
            exit(EXIT_FAILURE);
        }

        err = clFinish(commands); /// Aguarda até que o kernel tenha sido executado

        if(err){
            cout << "Error while running kernel (train). Error code: "<< err << endl;
        }

        auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
        long elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        
        /// Configuração dos argumentos para a realização do teste
        float TEST_LEARNING_RATE = -1;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buffer_X_test);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &buffer_y_test);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *) &NUM_TEST_OBSERVATIONS);
        err |= clSetKernelArg(kernel, 7, sizeof(float), (void *) &TEST_LEARNING_RATE);


        /// Comandos para execução do teste
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
  
        if (err != CL_SUCCESS){
            cout << "Error: Failed to execute kernel (test)! " << err << endl;
            exit(EXIT_FAILURE);
        }

        err = clFinish(commands); /// Aguarda até que o kernel tenha sido executado

        if(err){
            cout << "Error while running kernel (test). Error code: "<< err << endl;
        }

        /// Leitura das predições realizadas durante o teste
        err = clEnqueueReadBuffer(commands, buffer_predictions, CL_TRUE, 0, y_size, predictions, 0, NULL, NULL);

        if(err != CL_SUCCESS){
            cout << "Error while getting output" << endl;
        }

        /// Gravação das estatísticas de teste
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        ofstream outputFile;
        string outputFileName = OUTPUT_FILE_PREFIX+"output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test_@"+to_string(seed)+".txt"; /// Construção do título do arquivo de saída com as estatísticas de teste
        outputFile.open(outputFileName);
        outputFile << "epoch,accuracy,precision,recall,f1,cost" << endl; // Escreve o cabeçalho dos dados no arquivo de saída
        saveEpoch(-1, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS, -1);
        outputFile.close();

        ofstream timeFile;
        timeFile.open(OUTPUT_FILE_PREFIX+"time.txt", ios_base::app); // Concatena o tempo medido no final do arquivo
        timeFile << elapsed <<endl;
        timeFile.close();
        outputFile.close();
    }

    /// Liberação dos dados alocados utilizando o OpenCL
    clReleaseMemObject(buffer_X_train);
    clReleaseMemObject(buffer_y_train);
    clReleaseMemObject(buffer_X_test);
    clReleaseMemObject(buffer_y_test);
    clReleaseMemObject(buffer_weigths);
    clReleaseMemObject(buffer_predictions);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context); 

    /// Liberação dos dados alocados dinamicamente
    free(X_test);
    free(X_train);
    free(y_test);
    free(y_train);
    free(predictions);

}
