package com.yan.dl4j.Utils;

import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class MnistReadUtil {
    public static final String TRAIN_IMAGES_FILE = "src\\main\\resources\\data\\train-images.idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "src\\main\\resources\\data\\train-labels.idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "src\\main\\resources\\data\\t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "src\\main\\resources\\data\\t10k-labels.idx1-ubyte";

    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static double[][] getImages(String fileName) {
        try{
            return getImages(new FileInputStream(fileName));
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }
        return null;
    }

    public static double[][] getImages(InputStream inputStream) {
        double[][] x;
        try (BufferedInputStream bin = new BufferedInputStream(inputStream)) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new double[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    double[] element = new double[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
                        element[j] = bin.read();                                // 逐一读取像素值
                        // normalization
//                        element[j] = bin.read() / 255.0;
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }

    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return lables
     */
    public static double[] getLabels(String fileName) {
        try{
            return  getLabels(new FileInputStream(fileName));
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }
        return null;
    }

    public static double[] getLabels(InputStream inputStream) {
        double[] y;
        try (BufferedInputStream bin = new BufferedInputStream(inputStream)) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new double[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }


    public static void main(String[] args) {

        double[][] images = getImages(TEST_IMAGES_FILE);
        double[] labels = getLabels(TEST_LABELS_FILE);
        //double[] labels = getLabels(TRAIN_LABELS_FILE);
        try{
            for(int i=0;i<100;i++){
                drawGrayPicture(images[i],"0"+i+"l"+labels[i]+".jpg");
            }

        }catch (Exception e){
            e.printStackTrace();
        }


    }





    public static void drawGrayPicture(double[] pixelValues, String fileName) throws IOException {
        //double转int
        int[] res = new int[pixelValues.length];
        for(int i=0;i<pixelValues.length;i++){
            res[i] = new Double(pixelValues[i]).intValue();
        }
        //由数据集可以得知图片为28行28列的数据；
        int width = 28;
        int high = 28;
        BufferedImage bufferedImage = new BufferedImage(width, high, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < high; j++) {
                int pixel = 255 - res[i * high + j];
                int value = pixel + (pixel << 8) + (pixel << 16);   // r = g = b 时，正好为灰度
                bufferedImage.setRGB(i, j, value);
            }
        }
        ImageIO.write(bufferedImage, "JPEG", new File(fileName));
    }


    public static double[] getSizeBlackWhiteImg(File file,int width, int height) throws IOException {
        double[] result = null;
        if (!file.exists()) {
            System.out.println("图片不存在");
            return null;
        }
        BufferedImage bufImg = ImageIO.read(file);
        Image _img = bufImg.getScaledInstance(width, height, Image.SCALE_DEFAULT);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.drawImage(_img, 0, 0, null);
        graphics.dispose();
        int[] rgb = new int[3];
        result = new double[width*height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int r = image.getRGB(i, j) & 0xFFFFFF;
                rgb[0] = (r & 0xff0000) >> 16;
                rgb[1] = (r & 0xff00) >> 8;
                rgb[2] = (r & 0xff);
                int color = 255 - (int)(rgb[0]* 0.3 + rgb[1] * 0.59 + rgb[2] * 0.11);
                result[i*height+j] = color;
            }
        }
        return result;
    }
}
