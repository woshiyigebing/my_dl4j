package com.yan.dl4j.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class MyMathUtil {
    public static double MysigMoid(double value) {
        //Math.E=e;Math.Pow(a,b)=a^b
        double ey = Math.pow(Math.E, -value);
        return 1 / (1 + ey);
    }

    public static INDArray MysigMoid(INDArray value) {
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                for(int j =0;j<s[i].length;j++){
                    s[i][j] = MysigMoid(s[i][j]);
                }
            }
            return Nd4j.create(s);
        }else{
            double[] s = value.toDoubleVector();
            for(int i=0;i<s.length;i++){
                    s[i] = MysigMoid(s[i]);
            }
            return Nd4j.create(s);
        }
    }

    public static double Mytanh(double value) {
        double ex = Math.pow(Math.E, value);// e^x
        double ey = Math.pow(Math.E, -value);//e^(-x)
        double sinhx = ex-ey;
        double coshx = ex+ey;
        return sinhx/coshx;
    }
    public static INDArray Mytanh(INDArray value) {
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                for(int j =0;j<s[i].length;j++){
                    s[i][j] = Mytanh(s[i][j]);
                }
            }
            return Nd4j.create(s);
        }else{
            double[] s = value.toDoubleVector();
            for(int i=0;i<s.length;i++){
                s[i] = Mytanh(s[i]);
            }
            return Nd4j.create(s);
        }
    }

    public static double relu(double value) {
        return Math.max(0,value);
    }

    public static INDArray relu(INDArray value) {
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                for(int j =0;j<s[i].length;j++){
                    s[i][j] = relu(s[i][j]);
                }
            }
            return Nd4j.create(s);
        }else{
            double[] s = value.toDoubleVector();
            for(int i=0;i<s.length;i++){
                s[i] = relu(s[i]);
            }
            return Nd4j.create(s);
        }
    }

}
