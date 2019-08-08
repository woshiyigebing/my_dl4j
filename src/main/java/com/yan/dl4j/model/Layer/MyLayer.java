package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.ActivateMethod;
import com.yan.dl4j.model.WInit.Winit;
import com.yan.dl4j.model.WInit.XAVIER;

public class MyLayer implements Layer {

    private int number;
    private ActivateMethod activateMethod;
    private Winit winit;

    public MyLayer(int number,ActivateMethod method){
        this.number = number;
        this.activateMethod = method;
        this.winit = new XAVIER();
    }
    public MyLayer(int number,ActivateMethod method,Winit winit){
        this.number = number;
        this.activateMethod = method;
        this.winit = winit;
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

    @Override
    public Layer setWInit(Winit winit) {
        this.winit = winit;
        return this;
    }

    @Override
    public Winit getWinit() {
        return winit;
    }
}
