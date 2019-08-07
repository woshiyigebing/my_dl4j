package com.yan.dl4j.model.Activate;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SoftMax implements ActivateMethod {
    @Override
    public INDArray activate_forward(INDArray A) {
        return MyMathUtil.sotfmax(MyMathUtil.indArraysubMax(A));
    }

    @Override
    public INDArray activate_backward(INDArray DA, INDArray A) {
        return MyMathUtil.sotfmax_back(DA,A);
    }
}
