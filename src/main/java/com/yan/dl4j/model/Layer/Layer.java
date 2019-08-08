package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.ActivateMethod;
import com.yan.dl4j.model.WInit.Winit;

public interface Layer {
    int getNeuralNumber();
    ActivateMethod getActivateMethod();
    Layer setActivateMethod(ActivateMethod activate);
    Layer setWInit(Winit wInit);
    Winit getWinit();
}
