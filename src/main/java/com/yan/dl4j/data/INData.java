package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface INData {
    INDArray getX();
    INDArray getY();
    int getSize();
}
