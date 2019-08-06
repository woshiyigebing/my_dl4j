package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.ActivateMethod;
import com.yan.dl4j.model.Loss.LossMethod;

public class MyLastLayer extends MyLayer implements LastLayer {
    public MyLastLayer(int number,ActivateMethod method,LossMethod lossMethod){
        super(number,method);
        this.lossMethod = lossMethod;
    }
    private LossMethod lossMethod;
    @Override
    public LossMethod getLossMethod() {
        return lossMethod;
    }

    @Override
    public LastLayer setLossMethod(LossMethod lossMethod) {
        this.lossMethod = lossMethod;
        return this;
    }

}
