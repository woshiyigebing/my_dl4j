package com.yan.dl4j.model.Activate;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Relu implements ActivateMethod {

    @Override
    public INDArray activate_forward(INDArray A) {
        return MyMathUtil.relu(A);
    }

    @Override
    public INDArray activate_backward(INDArray DA, INDArray A) {
        return MyMathUtil.relu_back(DA);
    }
}
