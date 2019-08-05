package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.ActivateMethod;

public class MyLayer implements Layer {

    private int number;
    private ActivateMethod activateMethod;

    public MyLayer(int number,ActivateMethod method){
        this.number = number;
        this.activateMethod = method;
    }

    @Override
    public int getNeuralNumber() {
        return number;
    }


    @Override
    public ActivateMethod getActivateMethod() {
        return activateMethod;
    }

    @Override
    public Layer setActivateMethod(ActivateMethod activate) {
        this.activateMethod = activate;
        return this;
    }
}
