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
int NUM_ITERATIONS = 5;
float LEARNING_RATE = 0.001;            /// Taxa de Apredizado
string OUTPUT_FILE_PREFIX = "results/"; /// Prefixo do arquivo de saída

/**Source: http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl**/
long LoadOpenCLKernel(char const *path, char **buf)
{
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
void parsePixels(string pixels_str, float *pixels)
{
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
void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost)
{
    //cout << "done epoch #" <<epoch<<endl;
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for (int i = 0; i < size; i++)
    { ///calculando dados para matriz de confusão a partir do valor real e do valor predito
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

    accuracy = (tp + tn) / (tp + fp + fn + tn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = (2 * recall * precision) / (recall + precision);

    outputFile << epoch << "," << accuracy << "," << precision << "," << recall << "," << f1 << "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv)
{
    if (argc < 5)
    {
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA" << endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]);
    NUM_TEST_OBSERVATIONS = atof(argv[2]);
    NUM_EPOCHS = atof(argv[3]);
    LEARNING_RATE = atof(argv[4]);
    OUTPUT_FILE_PREFIX += argv[5];
    OUTPUT_FILE_PREFIX += "/";
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);
    
    float *X_train, *X_test, *weights;						
    float *y_train, *y_test;	

    X_train = (float *) malloc(NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float));
    X_test = (float *) malloc(NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float));    			
    y_train = (float *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));		 
    y_test = (float *)malloc(NUM_TEST_OBSERVATIONS * sizeof(float));		

    weights = (float *) malloc (NUM_FEATURES * sizeof(float));


    for(int i = 0; i < NUM_FEATURES; i++){									
        weights[i] = 0;
    }
    
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
	cout << "carregando treino" << endl;
    while(getline(trainFile, line)){    // Leitura do arquivo de entrada (treino)
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

    cout << "carregando teste "<< endl;
    while(getline(testFile, line)){    // Leitura do arquivo de entrada (teste)
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

    float predictions[NUM_TRAIN_OBSERVATIONS], cost;
    
    int error;
    /******** Initial Variables ********/
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceIdCount);

    std::vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

    int i = 0;
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

    if (error != CL_SUCCESS)
    {
        cout << "OpenCL error while getting info" << error << endl;
        exit(EXIT_FAILURE);
    }

    /******** End Initial Variables ********/

    if (error != CL_SUCCESS)
    {
        cout << "OpenCL error while allocating buffers" << error << endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        cout << "OpenCL buffers done" << endl;
    }

    cl_uint num_devices;
    error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    // Get the device IDs
    cl_device_id device[num_devices];
    error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
    //checkError(err, "Getting devices");
    printf("Number of devices: %d\n", num_devices);

    // Get device name
    error = clGetDeviceInfo(device[0], CL_DEVICE_NAME, sizeof(string), &string, NULL);
    //checkError(err, "Getting device name");
    printf("Device Name: %s\n", string);

    const cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformIds[0]), 0, 0};

    cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);

    if (error != CL_SUCCESS)
    {
        cout << "OpenCL error while creating context " << error << endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        cout << "--> OpenCL context created" << endl;
    }

    cl_command_queue commands = clCreateCommandQueue(context, device[0], 0, &error);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    char *KernelSource;
    long lFileSize;

    lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);

    if (lFileSize < 0L)
    {
        cout << "File read failed" << endl;
        return 1;
    }
    else
    {
        cout << "Got source" << endl;
    }

    cout << KernelSource << endl;

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &error);

    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                              buffer, &len);

        printf("%s\n", buffer);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "logreg", &error);

    if (!kernel || error != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        cout << "error code " << error << endl;
        exit(1);
    }
    else
    {
        cout << "Kernel created" << endl;
    }

    /******** Buffers *******/
    cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float), nullptr, &error);
    
    cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TRAIN_OBSERVATIONS * sizeof(float), nullptr, &error);

    //cl_mem buffer_X_test = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float), nullptr, &error);
    
    //cl_mem buffer_y_test = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_TEST_OBSERVATIONS * sizeof(float), nullptr, &error);

    cl_mem buffer_weigths =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_FEATURES * sizeof(float), nullptr, &error);
    
    cl_mem buffer_predictions =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_TRAIN_OBSERVATIONS * sizeof(float), nullptr, &error);

    cl_mem buffer_costs =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_TRAIN_OBSERVATIONS * sizeof(float), nullptr, &error);

    cl_mem buffer_NUM_FEATURES =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), nullptr, &error);

    cl_mem buffer_NUM_TRAIN_OBSERVATIONS =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), nullptr, &error);

    cl_mem buffer_LEARNING_RATE =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &error);


    if(!buffer_X_train                  || 
        !buffer_y_train                 ||
        !buffer_weigths                 ||
        !buffer_predictions             ||
        !buffer_costs                   ||
        !buffer_NUM_FEATURES            ||
        !buffer_NUM_TRAIN_OBSERVATIONS  ||
        !buffer_LEARNING_RATE
        ){
        printf("Error: Failed to allocate device memory!\n");
        //exit(1);
    }

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_X_train);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffer_y_train);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buffer_weigths);
    error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&buffer_predictions);
    error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&buffer_costs);
    error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&buffer_NUM_FEATURES);
    error |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&buffer_NUM_TRAIN_OBSERVATIONS);
    error |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&buffer_LEARNING_RATE);


    if (error != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", error);
        //exit(1);
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 32;
    localWorkSize[1] = 32;
    globalWorkSize[0] = 1024;
    globalWorkSize[1] = 1024;

    /******** End Buffers ********/

    error = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 
   0, NULL, NULL);

   if (error != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", error);
       //exit(1);
   }

   clReleaseMemObject(buffer_X_train);
   clReleaseMemObject(buffer_y_train);
   clReleaseMemObject(buffer_weigths);
   clReleaseMemObject(buffer_predictions);
   clReleaseMemObject(buffer_costs);
   clReleaseMemObject(buffer_NUM_FEATURES);
   clReleaseMemObject(buffer_NUM_TRAIN_OBSERVATIONS);
   clReleaseMemObject(buffer_LEARNING_RATE);

   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);
   
   free(X_test);
   free(X_train);
   free(y_test);
   free(y_train);
   free(weights);

}
