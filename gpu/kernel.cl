__kernel void logreg(
    __global float* X_train,
    __global float* y_train,
    __global float* weights,
    __global float* predictions,
    __global float* partialGradients,
    int NUM_FEATURES,
    int NUM_TRAIN_OBSERVATIONS,
    float LEARNING_RATE,
    int NUM_EPOCHS){
    
   const int localId = get_global_id(0);
   const int NUM_WORK_ITEMS = get_global_size(0);
   const int localExamplesStart = NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS * localId;

   for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){

        for(int k = 0; k < NUM_FEATURES; k++){
                partialGradients[localId*NUM_FEATURES + k] = 0;
                weights[k] = 0;
        }

        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS; i++){
            float z = 0;

            for (int j = 0; j < NUM_FEATURES; j++){
                z += X_train[(localExamplesStart + i) * NUM_FEATURES + j] * weights[j];
            }

            predictions[localExamplesStart + i] =  1.0/(1.0+exp(-z));
            z = y_train[localExamplesStart+i] - 1.0/(1.0+exp(-z));

            for(int j = 0; j < NUM_FEATURES; j++){
                partialGradients[localId * NUM_FEATURES + j] += X_train[(localExamplesStart+i)*NUM_FEATURES + j] * z;
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        for(int offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1){ 
            for (int k = 0; k < NUM_FEATURES; k++)
                partialGradients[localId*NUM_FEATURES + k] += partialGradients[(localId+offset)*NUM_FEATURES + k]; 
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);


        if(localId < NUM_FEATURES){
            weights[localId] += (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * partialGradients[localId];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
   }    
}