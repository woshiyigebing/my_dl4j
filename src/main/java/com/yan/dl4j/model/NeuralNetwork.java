package com.yan.dl4j.model;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.omg.PortableInterceptor.INACTIVE;

import java.util.Arrays;

public class NeuralNetwork implements model  {

    private double learningrate=0.01;

    private int iteration = 1000;

    //4个神经元 3个特征值 20个数据
    //w = 4*3  x = 3*20 y = 4*20   =》z1 = 4*20;
    private INDArray Network_1LAYER_W = Nd4j.ones(4, 3);
    private INDArray Network_1LAYER_B = Nd4j.ones(4, 1);
    //1个神经元 4个特征值 1个数据
    //w = 1*4 x = 4*20 y = 1*20 => z2 = 1*20;
    private INDArray Network_2LAYER_W = Nd4j.create(new float[]{2, 2, 2, 2}, new int[]{1, 4});
    private INDArray Network_2LAYER_B = Nd4j.create(new float[]{2}, new int[]{1, 1});

    public static void main(String[] args) {
        INDArray my_Network_2LAYER_W = Nd4j.rand(4, 20);
        System.out.println(my_Network_2LAYER_W);
       my_Network_2LAYER_W = MyMathUtil.MysigMoid(my_Network_2LAYER_W);
        System.out.println(my_Network_2LAYER_W);
    }


    @Override
    public void train(TrainData data) {

        //向前传播
        INDArray X = data.getX();
        INDArray Y = data.getY();
        INDArray Z1 = getNetwork_1LAYER_W().mmul(X).add(getNetwork_1LAYER_B());
        INDArray A1 = MyMathUtil.Mytanh(Z1);
        INDArray Z2 = getNetwork_2LAYER_W().mmul(A1).add(getNetwork_2LAYER_B());
        INDArray A2 = MyMathUtil.Mytanh(Z2);
        //向后传播

        //Loss函数

        //梯度下降

    }

    @Override
    public INDArray predict(INDArray x) {
        return null;
    }

    @Override
    public double getLearningrate() {
        return learningrate;
    }

    @Override
    public int getIteration() {
        return iteration;
    }

    public INDArray getNetwork_1LAYER_W() {
        return Network_1LAYER_W;
    }

    public INDArray getNetwork_1LAYER_B() {
        return Network_1LAYER_B;
    }

    public INDArray getNetwork_2LAYER_W() {
        return Network_2LAYER_W;
    }

    public INDArray getNetwork_2LAYER_B() {
        return Network_2LAYER_B;
    }
}
