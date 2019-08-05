package com.yan.dl4j.controller;

import com.yan.dl4j.Utils.MnistReadUtil;
import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.model.Activate.Relu;
import com.yan.dl4j.model.Layer.MyLayer;
import com.yan.dl4j.model.Layer.SotfMaxCrossEntropyLastLayer;
import com.yan.dl4j.model.NetWork.DeepNeuralNetWork;
import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

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
        TrainData data = new MyTrainData(X_I,Y_I,128);
        model nk = new DeepNeuralNetWork(28*28)
                .addLayer(new MyLayer(1000,new Relu()))
                .addLastLayer(new SotfMaxCrossEntropyLastLayer(10));
//        List<Integer> LARYER = new ArrayList<>();
//        LARYER.add(28*28);
//        LARYER.add(1000);
//        LARYER.add(10);
//        model nk = new DeepNeuralNetWork(LARYER,"CrossEntropy");
        nk.train(data);
        return "success";
    }

}
