package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.ActivateMethod;

public interface Layer {
    int getNeuralNumber();
    ActivateMethod getActivateMethod();
    Layer setActivateMethod(ActivateMethod activate);
}
