package com.yan.dl4j.model;

import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class LinearRegressionDL4J implements model{

    //用于打印坐标系点
    private INDArray x;
    private INDArray y;

    //初始化斜率
    public double w = 0;

    //初始化截距
    public double b = 0;

    //初始化学习速度（梯度下降速度）
    public double learningrate = 0.01d;

    public static int iteration = 1000;


    //初始化加载图像数据
    public LinearRegressionDL4J() {
    }


    public void fitBGD(INDArray x, INDArray y)
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
        INDArray yc = x.mul(this.getW()).add(this.getB());
        return yc;
    }

    //误差值
    public double calc_error(INDArray x, INDArray y)
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


    public double getW()
    {
        return w;
    }

    public void setW(double w)
    {
        this.w = w;
    }

    public double getB()
    {
        return b;
    }

    public void setB(double b)
    {
        this.b = b;
    }

    public static List<Double> doublesData(List<Double> list)
    {
        DecimalFormat df = new DecimalFormat("#.0000000000");
        List<Double> xLists = new ArrayList<>();
        for (int i = 0; i < list.size(); i++)
        {
            String str = df.format(list.get(i));
            xLists.add(Double.valueOf(str));
        }
        return xLists;
    }
}
