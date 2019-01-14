__kernel void logreg(
    __global float* X_train,
    __global float* y_train,
    __global float* weights,
    __global float* predictions,
    __global float* costs,
    int NUM_FEATURES,
    int NUM_TRAIN_OBSERVATIONS,
    float LEARNING_RATE)
{
    for(int j= 0; j < 100; j++){
   int tx = get_global_id(0); 

   // predictions
   float value = 0;

   for (unsigned int i = 0; i < NUM_FEATURES; i++) {
      value += X_train[tx * NUM_FEATURES + i] * weights[i];
   }

    predictions[tx] = 1.0 / (1.0 + exp(-value)); // sigmoid function

   // update weigths
   float sum = 0;    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        float h_xi = predictions[i];
        sum += (h_xi - y_train[i]) * X_train[tx * NUM_FEATURES + i];
    }

    float gradient = (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * sum;

    weights[tx] -= gradient;

   // cost
   float cost = 0;
   float h_xi;

   h_xi = predictions[tx];
   float p1 = y_train[tx] * log(h_xi);
   float p2 = (1-y_train[tx]) * log(1-h_xi);

   costs[tx] = (-p1-p2);
    }
}