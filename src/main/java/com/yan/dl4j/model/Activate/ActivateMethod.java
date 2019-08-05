package com.yan.dl4j.model.Activate;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActivateMethod {
    INDArray activate_forward(INDArray A);
    INDArray activate_backward(INDArray DA, INDArray A);
}
