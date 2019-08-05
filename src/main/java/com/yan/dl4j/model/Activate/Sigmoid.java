package com.yan.dl4j.model.Activate;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Sigmoid implements ActivateMethod {

    @Override
    public INDArray activate_forward(INDArray A) {
        return MyMathUtil.MysigMoid(A);
    }

    @Override
    public INDArray activate_backward(INDArray DA, INDArray A) {
        return DA.mul(A).mul(Nd4j.ones(A.shape()).sub(A));
    }
}
