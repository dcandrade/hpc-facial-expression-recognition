__kernel void logreg(
    __global float* X,
    __global float* y,
    __global float* weights,
    __global float* predictions,
    __global float* partialGradients,
    int NUM_FEATURES,
    int NUM_OBSERVATIONS,
    float LEARNING_RATE,
    int NUM_EPOCHS)
{
    
   const int tx = get_global_id(0);
   const int NUM_WORK_ITEMS = get_global_size(0);
   const int localExamplesStart = NUM_OBSERVATIONS/NUM_WORK_ITEMS * tx;

   for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){

        for(int k = 0; k < NUM_FEATURES; k++){
                partialGradients[tx*NUM_FEATURES + k] = 0.0f;
                weights[k] = 0.0f;
        }

        for (int i = 0; i < NUM_OBSERVATIONS/NUM_WORK_ITEMS; i++){
            float z = 0.0f;

            for (int j = 0; j < NUM_FEATURES; j++){
                z += X[(localExamplesStart + i) * NUM_FEATURES + j] * weights[j];
            }

            predictions[localExamplesStart + i] =  1.0/(1.0+exp(-z));
            z = y[localExamplesStart+i] - 1.0/(1.0+exp(-z));

            for(int j = 0; j < NUM_FEATURES; j++){
                partialGradients[tx * NUM_FEATURES + j] += X[(localExamplesStart+i)*NUM_FEATURES + j] * z;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1){ 
            for (int k = 0; k < NUM_FEATURES; k++)
                partialGradients[tx*NUM_FEATURES + k] += partialGradients[(tx+offset)*NUM_FEATURES + k]; 
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(tx < NUM_FEATURES){
            weights[tx] += (LEARNING_RATE/NUM_OBSERVATIONS) * partialGradients[tx];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
   }
}