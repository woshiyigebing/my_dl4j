package com.yan.dl4j.model;

import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class LinearRegressionDL4J implements model{

    //初始化斜率
    private double w = 0;

    //初始化截距
    private double b = 0;

    //初始化学习速度（梯度下降速度）
    private double learningrate = 0.01d;

    private int iteration = 1000;


    //初始化加载图像数据
    public LinearRegressionDL4J() {
    }


    private void fitBGD(INDArray x, INDArray y)
    {
        double wt = this.getW();
        double bt = this.getB();
        double learningrate = this.getLearningrate();

        INDArray diff = y.dup().sub(x.mul(wt)).sub(bt);
        wt = wt + diff.dup().muli(x).sumNumber().doubleValue() / x.length() * 2 * learningrate;
        bt = bt + diff.sumNumber().doubleValue() / x.length() * 2 * learningrate;


        double loss = Math.pow(calc_error(x, y), 2);

        System.out.println("w->" + wt);
        System.out.println("b->" + bt);
        System.out.println("loss->" + loss);


        this.setW(wt);
        this.setB(bt);
    }

    @Override
    public void train(TrainData data) {
        for (int i = 0; i < iteration; i++)
        {
            fitBGD(data.getX(), data.getY());
            System.out.println("i->"+i);
            System.out.println("\n");
        }
    }

    //目标函数
    public INDArray predict(INDArray x)
    {
        //wx+b
        return x.mul(this.getW()).add(this.getB());
    }

    //误差值
    private double calc_error(INDArray x, INDArray y)
    {
        //y-wx-b
        INDArray yc = predict(x);
        INDArray error = y.sub(yc);
        return error.sumNumber().doubleValue();
    }


    public double getLearningrate()
    {
        return learningrate;
    }


    private double getW()
    {
        return w;
    }

    private void setW(double w)
    {
        this.w = w;
    }

    private double getB()
    {
        return b;
    }

    private void setB(double b)
    {
        this.b = b;
    }

    public int getIteration() {
        return iteration;
    }

}
