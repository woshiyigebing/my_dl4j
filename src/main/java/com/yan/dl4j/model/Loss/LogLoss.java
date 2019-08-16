package com.yan.dl4j.model.Loss;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LogLoss implements LossMethod {

    @Override
    public INDArray LossBackward(INDArray A, INDArray Y) {
        return A.sub(Y).div(A.mul(Nd4j.ones(A.shape()).sub(A)));
    }

    @Override
    public double LossForward(INDArray A, INDArray Y) {
        INDArray logb =  (Nd4j.ones(Y.shape()).sub(Y)).mul(MyMathUtil.Log(Nd4j.ones(A.shape()).sub(A))).add(MyMathUtil.Log(A).mul(Y));
        return  (0-logb.sumNumber().doubleValue())/Y.shape()[1];
    }
}
