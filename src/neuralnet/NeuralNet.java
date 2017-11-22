package neuralnet;

/**
 *
 * @author blakk
 */

import java.util.Scanner;
import java.util.ArrayList;


public class NeuralNet {
    static String version = "1.0.0";
    public Network net = new Network(2,3,1);
    
    public static void main(String[] args) {
        Interface face = new Interface();
        face.setTitle("Neural Network 9001 V"+version);
        face.setVisible(true);
        
        //Scanner sc = new Scanner(System.in);
        //2:3:1 network for boolean logic
        
        //layer 0 inputs
        /* manual inputs
        int inputSize = 2;
        double[] inputLayer = new double[inputSize];
        //input inputs
        for(int in = 0; in < inputSize; in++){
            System.out.println("Input node "+in);
            inputLayer[in] = sc.nextDouble();
        }
        */
        
        //random network
        
        //custom network
        /*double[][] iwHidden = {{0,0,0},{0,0,0}}; //2x3 of 0
        double[][] iwOutput = {{0},{0},{0}}; //3x1 of 0        
        Matrix hiddenWeightMatrix = new Matrix(iwHidden);
        Matrix outputWeightMatrix = new Matrix(iwOutput);        
        Network net = new Network(hiddenWeightMatrix, outputWeightMatrix, 2,3,1);
        */
        
        /* ### WORKING CLI NETWORK
        Network net = new Network(2,3,1);
        
        //training for an AND gate
        ArrayList<double[]> ANDin = new ArrayList();
        ArrayList<Double> ANDout = new ArrayList();
        
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                double[] inputArray = {i,j};
                double output = i & j;
                //System.out.println(output);
                ANDin.add(inputArray);
                ANDout.add(output);
            }
        }
        System.out.println("Turns:");
        for(int turns = sc.nextInt();turns > 0; turns--){
            if(ANDin.size() == ANDout.size()){
                for (int set = 0; set < ANDin.size(); set ++){
                    net.train(ANDin.get(set),ANDout.get(set));
                }
            }
        }
        */
        
        
    }    
}
