package com.yan.dl4j.controller;

import com.yan.dl4j.Utils.MnistReadUtil;
import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.INData;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.model.Activate.Relu;
import com.yan.dl4j.model.Activate.Sigmoid;
import com.yan.dl4j.model.Activate.SoftMax;
import com.yan.dl4j.model.Layer.MyLastLayer;
import com.yan.dl4j.model.Layer.MyLayer;
import com.yan.dl4j.model.Layer.SotfMaxCrossEntropyLastLayer;
import com.yan.dl4j.model.Loss.LogLoss;
import com.yan.dl4j.model.Loss.MSE;
import com.yan.dl4j.model.NetWork.DeepNeuralNetWork;
import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.SoftmaxCrossEntropyLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/mnist")
public class MnistController {
    public static final ClassPathResource TRAIN_IMAGES_FILE = new ClassPathResource("data/train-images.idx3-ubyte");
    public static final ClassPathResource TRAIN_LABELS_FILE = new ClassPathResource("data/train-labels.idx1-ubyte");
    public static final ClassPathResource TEST_IMAGES_FILE = new ClassPathResource("data/t10k-images.idx3-ubyte");
    public static final ClassPathResource TEST_LABELS_FILE = new ClassPathResource("data/t10k-labels.idx1-ubyte");

    @GetMapping(value = "train")
    public String train() throws Exception{
        double[][] images = MnistReadUtil.getImages(TRAIN_IMAGES_FILE.getInputStream());
        double[] labels = MnistReadUtil.getLabels(TRAIN_LABELS_FILE.getInputStream());
        INDArray X = Nd4j.create(images);  //60000,784
        INDArray Y = Nd4j.create(labels).transpose(); //60000,1
        INDArray X_I = MyMathUtil.Normalization(X);
        INDArray Y_I = MyMathUtil.ONEHOT(Y);//60000,10
        TrainData data = new MyTrainData(X_I,Y_I,8192);
        model nk = new DeepNeuralNetWork(28*28)
                .addLayer(new MyLayer(1000,new Relu()))
                .addLastLayer(new SotfMaxCrossEntropyLastLayer(10)).setIteration(100).setLearningrate(0.006);
//        List<Integer> LARYER = new ArrayList<>();
//        LARYER.add(28*28);
//        LARYER.add(1000);
//        LARYER.add(10);
//        model nk = new DeepNeuralNetWork(LARYER,"CrossEntropy");
        nk.train(data);
        double[][] t_images = MnistReadUtil.getImages(TEST_IMAGES_FILE.getInputStream());
        double[] t_labels = MnistReadUtil.getLabels(TEST_LABELS_FILE.getInputStream());
        INDArray X_t = MyMathUtil.Normalization(Nd4j.create(t_images));
        INDArray Y_t = MyMathUtil.ONEHOT(Nd4j.create(t_labels).transpose());
        TrainData data_t = new MyTrainData(X_t,Y_t);
        INDArray X_P = nk.predict(data_t.getX());
        System.out.println(scord(X_P,data_t.getY()));
        return "success";
    }

    @GetMapping(value = "test")
    public String test() {
        double[][] matrixDouble = new double[][]{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3},
                {4, 4, 4},
                {5, 5, 5},
                {6, 6, 6},
                {7, 7, 7},
                {8, 8, 8},
                {9, 9, 9},
        };
        INDArray X = Nd4j.create (matrixDouble);  //9,3
        INDArray Y = Nd4j.create (matrixDouble);  //9,3
        TrainData data = new MyTrainData(X,Y,2);
        List<INData> data1 =  data.getBatchList();
        return "success";
    }

    public float scord(INDArray value,INDArray Y) {
        int res = 0;
        int sum = 0;
            double[][] s = value.transpose().toDoubleMatrix();
            double[][] Ys = Y.transpose().toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                double Max = -1;
                int order = -1;
                for(int j =0;j<s[i].length;j++){
                    order = Max>s[i][j]?order:j;
                    Max = Max>s[i][j]?Max:s[i][j];
                }
                if(order>0&&new Double(Ys[i][order]).intValue()==1){
                    res++;
                }
                sum++;
            }
            if(sum>0){
                return ((float)res/sum)*100;
            }else{
                return 0;
            }
    }

}
