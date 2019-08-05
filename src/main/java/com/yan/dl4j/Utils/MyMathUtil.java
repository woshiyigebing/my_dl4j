package com.yan.dl4j.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.DoubleFunction;

public class MyMathUtil {
    public static double Epow(double x){
        return Math.pow(Math.E, x);//e^x
    }

    public static INDArray Epow(INDArray value){
        return FUN_IND(value,v->Epow(v));
    }

    public static double Normalization(double value,double Max){
        return value/Max;
    }

    public static INDArray Normalization(INDArray value){
        double my_Max = 1;
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            for(double[] si:s){
                for(double sj:si){
                    my_Max = my_Max>sj?my_Max:sj;
                }
            }
        }else{
            double[] s = value.toDoubleVector();
            for(double si:s){
                my_Max = my_Max>si?my_Max:si;
            }
        }
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                for(int j =0;j<s[i].length;j++){
                    s[i][j] = s[i][j]/my_Max;
                }
            }
            return Nd4j.create(s);
        }else{
            double[] s = value.toDoubleVector();
            for(int i=0;i<s.length;i++){
                s[i] = s[i]/my_Max;
            }
            return Nd4j.create(s);
        }
    }

    public static INDArray ONEHOT(INDArray value){
        if(value.isColumnVector()){
            int[] s = value.toIntVector();
            int Max = 0;
            for(int si:s){
                Max = Max>si?Max:si;
            }
            double[][] one_hot_res= new double[s.length][Max+1];
            for(int i=0;i<s.length;i++){
                int val =  s[i];
                for(int j=0;j<Max+1;j++){
                    if(val==j){
                        one_hot_res[i][j] = 1;
                    }else{
                        one_hot_res[i][j] = 0;
                    }
                }
            }

            return Nd4j.create(one_hot_res);
        }
        return null;
        }

    public static INDArray FUN_IND(INDArray value, DoubleFunction<Double> doubleFunction){
        if(value!=null){
            if(value.shape()[0]>1){
                double[][] s = value.toDoubleMatrix();
                for(int i=0;i<s.length;i++){
                    for(int j =0;j<s[i].length;j++){
                        s[i][j] = doubleFunction.apply(s[i][j]);
                    }
                }
                return Nd4j.create(s);
            }else{
                double[] s = value.toDoubleVector();
                for(int i=0;i<s.length;i++){
                    s[i] = doubleFunction.apply(s[i]);
                }
                return Nd4j.create(s);
            }
        }
        return null;
    }

    public static double MysigMoid(double value) {
        //Math.E=e;Math.Pow(a,b)=a^b
        double ey = Math.pow(Math.E, -value);
        return 1 / (1 + ey);
    }

    public static INDArray MysigMoid(INDArray value) {
        return FUN_IND(value,v->MysigMoid(v));
    }

    public static double Mytanh(double value) {
        double ex = Math.pow(Math.E, value);// e^x
        double ey = Math.pow(Math.E, -value);//e^(-x)
        double sinhx = ex-ey;
        double coshx = ex+ey;
        return sinhx/coshx;
    }
    public static INDArray Mytanh(INDArray value) {
        return FUN_IND(value,v->Mytanh(v));
    }

    public static double relu(double value) {
        return Math.max(0,value);
    }

    public static INDArray relu(INDArray value) {
        return FUN_IND(value,v->relu(v));
    }

    public static double relu_back(double value) {
        if(value>0){
            return value;
        }else{
            return 0;
        }
    }

    public static INDArray relu_back(INDArray value) {
        return FUN_IND(value,v->relu_back(v));
    }

    public static double Log(double value) {
        return Math.log(value);
    }

    public static INDArray Log(INDArray value) {
        return FUN_IND(value,v->Log(v));
    }

    public static INDArray sotfmax(INDArray A){
        if(A!=null){
            A = MyMathUtil.Epow(A);  //A: 10,128
            INDArray sum_A = Nd4j.ones(1,A.shape()[0]).mmul(A); //1,128
            if(A.shape()[0]>1){
                double[][] A_s = A.toDoubleMatrix();
                double[] SUM_A_s = sum_A.toDoubleVector();
                for(int i=0;i<A_s.length;i++){
                    for(int j =0;j<A_s[i].length;j++){
                        A_s[i][j] = A_s[i][j]/SUM_A_s[j];
                    }
                }
                return Nd4j.create(A_s);
            }else{
                return Nd4j.ones(A.shape());
            }
        }
        return null;
    }
}
