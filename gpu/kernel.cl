/**
 * @file kernel.cl
 * @author Daniel Andrade e Gabriel Gomes
 * @brief Reconhecedor de Expressões Faciais através de Regressão Logística - Kernel GPU
 * @version 1.0
 * @date 2019-05-02
 * 
 * @copyright Copyright (c) 2019
 * 
 */
__kernel void logreg(
    __global float* X,
    __global float* y,
    __global float* weights,
    __global float* predictions,
    __local float* partialGradients,
    int NUM_FEATURES,
    int NUM_OBSERVATIONS,
    float LEARNING_RATE,
    int NUM_EPOCHS)
{
    
   const int tx = get_local_id(0); /// Número identificador da thread (work-item) atual
   const int NUM_WORK_ITEMS = get_local_size(0); /// Quantidade de threads (work-items) na qual o kernel está sendo executado
   const int startIndex = NUM_OBSERVATIONS/NUM_WORK_ITEMS * tx; /// Índice da primeira amostra pela qual a thread (work-item) atual é responsável (cada thread é responsável por NUM_OBSERVATIONS/NUM_WORK_ITEMS amostras)
   const int startFeatureIndex = NUM_FEATURES/NUM_WORK_ITEMS * tx;

    if(LEARNING_RATE > 0){ /// Executa no modo de treinamento
        /// Inicialização dos pesos
        for(int j = 0; j < NUM_FEATURES/NUM_WORK_ITEMS; j++){
            weights[startFeatureIndex + j] = 0;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){
            /// Reinicialização dos valores do gradiente
            for(int k = 0; k < NUM_FEATURES; k++){
                partialGradients[tx*NUM_FEATURES + k] = 0;
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);

            /// Cálculo das hipóteses e dos gradientes parciais
            for (int i = 0; i < NUM_OBSERVATIONS/NUM_WORK_ITEMS; i++){
                float z = 0;

                for (int j = 0; j < NUM_FEATURES; j++){
                    z += (X[(startIndex + i) * NUM_FEATURES + j] * weights[j]);
                }

                z = y[startIndex+i] - 1.0/(1.0+exp(-z));

                for(int j = 0; j < NUM_FEATURES; j++){
                    partialGradients[tx * NUM_FEATURES + j] += X[(startIndex+i)*NUM_FEATURES + j] * z;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE); /// Barreira necessária porque a próxima etapa precisa de todos os gradientes parciais calculados

            /// Redução dos gradientes parciais
            for(int offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1){ 
                for (int k = 0; k < NUM_FEATURES; k++)
                    partialGradients[tx*NUM_FEATURES + k] += partialGradients[(tx+offset)*NUM_FEATURES + k]; 
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            barrier(CLK_LOCAL_MEM_FENCE); /// Barreira necessária porque os pesos só podem ser atualizados quando todos os gradientes tiverem sido reduzidos

            /// Atualização dos pesos utilizando os gradientes
            for(int j = 0; j < NUM_FEATURES/NUM_WORK_ITEMS; j++){
                weights[startFeatureIndex + j] += (LEARNING_RATE/NUM_OBSERVATIONS) * partialGradients[startFeatureIndex + j];
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE); /// Barreira necessária porque a próxima época só pode inciar quando todos os pesos já tiverem sido atualizados na memória global
        }
    }else{ /// Executa no modo de teste
        /// Cálculo das predições
        for (int i = 0; i < NUM_OBSERVATIONS/NUM_WORK_ITEMS; i++){
            float z = 0;

            for (int j = 0; j < NUM_FEATURES; j++){
                z += (X[(startIndex + i) * NUM_FEATURES + j] * weights[j]);
            }

            predictions[startIndex + i] =  1.0/(1.0+exp(-z));
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}   
