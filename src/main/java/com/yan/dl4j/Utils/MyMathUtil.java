package com.yan.dl4j.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class MyMathUtil {
    public static double MysigMoid(double value) {
        //Math.E=e;Math.Pow(a,b)=a^b
        double ey = Math.pow(Math.E, -value);
        double result = 1 / (1 + ey);
        return result;
    }

    public static INDArray MysigMoid(INDArray value) {
        double[][] s = value.toDoubleMatrix();
        for(int i=0;i<s.length;i++){
            for(int j =0;j<s[i].length;j++){
                s[i][j] = MysigMoid(s[i][j]);
            }
        }
        return Nd4j.create(s);
    }

    public static double Mytanh(double value) {
        double ex = Math.pow(Math.E, value);// e^x
        double ey = Math.pow(Math.E, -value);//e^(-x)
        double sinhx = ex-ey;
        double coshx = ex+ey;
        double result = sinhx/coshx;
        return result;
    }
    public static INDArray Mytanh(INDArray value) {
        double[][] s = value.toDoubleMatrix();
        for(int i=0;i<s.length;i++){
            for(int j =0;j<s[i].length;j++){
                s[i][j] = Mytanh(s[i][j]);
            }
        }
        return Nd4j.create(s);
    }

}
