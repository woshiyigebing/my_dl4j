package com.yan.dl4j.model.Loss;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MSE implements  LossMethod {
    @Override
    public INDArray LossBackward(INDArray A, INDArray Y) {
        return A.sub(Y);
    }

    @Override
    public double LossForward(INDArray A, INDArray Y) {
        return (Y.sub(A).mmul(Y.sub(A).transpose()).sumNumber().doubleValue())/Y.shape()[1];
    }
}
