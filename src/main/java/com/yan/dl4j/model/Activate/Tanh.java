package com.yan.dl4j.model.Activate;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Tanh implements ActivateMethod {
    @Override
    public INDArray activate_forward(INDArray A) {
        return MyMathUtil.Mytanh(A);
    }

    @Override
    public INDArray activate_backward(INDArray DA, INDArray A) {
        return DA.mul(Nd4j.ones(A.shape()).sub(A.mul(A)));
    }
}
