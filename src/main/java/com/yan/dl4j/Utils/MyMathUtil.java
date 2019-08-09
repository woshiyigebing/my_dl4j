package com.yan.dl4j.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.math.BigDecimal;
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

    public static double MaxValue(INDArray value){
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            double my_Max=s[0][0];
            for(double[] si:s){
                for(double sj:si){
                    my_Max = Math.max(my_Max,sj);
                }
            }
            return my_Max;
        }else{
            double[] s = value.toDoubleVector();
            double my_Max=s[0];
            for(double si:s){
                my_Max = Math.max(my_Max,si);
            }
            return my_Max;
        }
    }

    public static INDArray Normalization(INDArray value){
        double my_Max = MaxValue(value);
        return FUN_IND(value,s->s/my_Max);
    }

    public static INDArray indArraysubMax(INDArray value){
        if(value!=null){
            if(value.shape()[0]>1){
                double[][] s = value.transpose().toDoubleMatrix();
                for(int i=0;i<s.length;i++){
                    double Max = s[i][0];
                    for(int j =0;j<s[i].length;j++){
                        Max = Math.max(Max,s[i][j]);
                    }
                    for(int j =0;j<s[i].length;j++){
                        s[i][j] = new BigDecimal(s[i][j]).subtract(new BigDecimal(Max)).doubleValue();
                    }
                }
                return Nd4j.create(s).transpose();
            }else{
                double[] s = value.toDoubleVector();
                for(int i=0;i<s.length;i++){
                    s[i] = 0;
                }
                return Nd4j.create(s);
            }
        }
        return null;
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
                double[][] A_s = A.transpose().toDoubleMatrix(); //128 10
                double[] SUM_A_s = sum_A.toDoubleVector();
                for(int i=0;i<A_s.length;i++){
                    for(int j =0;j<A_s[i].length;j++){
                        A_s[i][j] = A_s[i][j]/SUM_A_s[i];
                    }
                }
                return Nd4j.create(A_s).transpose();
            }else{
                return Nd4j.ones(A.shape());
            }
        }
        return null;
    }

    //公式：
    public static INDArray sotfmax_back(INDArray DA,INDArray A){
        if(A!=null){
            double[][] da = DA.transpose().toDoubleMatrix();
            double[][] a = A.transpose().toDoubleMatrix();
            double[][] res = new double[da.length][da[0].length];
                for(int i=0;i<da.length;i++){
                    int i_order = 0;
                    for(int j=0;j<da[i].length;j++){
                        if(da[i][j]!=0){i_order = j;}
                    }
                    for(int j=0;j<da[i].length;j++){
                        if(j==i_order){
                            res[i][j] = a[i][j]*(1-a[i][j]);
                        }else{
                            res[i][j] = -a[i][j]*(a[i][i_order]);
                        }
                    }
                }
                return Nd4j.create(res).transpose();
        }
        return null;
    }
}
