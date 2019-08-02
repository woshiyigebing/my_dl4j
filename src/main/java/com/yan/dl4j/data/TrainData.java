package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public interface TrainData extends INData{
     List<INData> getBatchList();
}
