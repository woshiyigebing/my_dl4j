package com.yan.dl4j.model.Loss;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface LossMethod {
    INDArray LossBackward(INDArray A,INDArray Y);
    double LossForward(INDArray A,INDArray Y);
}
