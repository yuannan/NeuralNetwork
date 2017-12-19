/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;
import java.util.ArrayList;

/**
 *
 * @author blakk
 */


public class Network {
    //settings up tracker
    int resultsSize = 100;
    int resultsCount = 0;
    ArrayList<ArrayList> results = new ArrayList(resultsSize);
    boolean debug = false;
    boolean trained = false;
    
    //init constants
    Matrix hiddenWeights, outputWeights; //weights
    Matrix DhiddenWeights, DoutputWeights; //delta weights
    Matrix inputLayer, hiddenIn, hiddenOut, outputIn, outputOut; //layer values
    int in, hidden, out;
    
    public double[] output;
    public double[] error;
    
    //init of a 3 layer network with weights
    public Network(Matrix inputToHidden, Matrix hiddenToOut, int inSize, int hiddenSize, int outSize){
        //setting network constants
        this.initConstants(inSize, hiddenSize, outSize);
        hiddenWeights = inputToHidden;
        outputWeights = hiddenToOut;
        
        //validating network weight sizes
        if(hiddenWeights.rows != in || hiddenWeights.cols != hidden) System.out.println("Inproper hidden weight size!");
        if(outputWeights.rows != hidden  || outputWeights.cols != out) System.out.println("Inproper output weight size!");
    }
    
    //init of a 3 layer network with random weights
    public Network(int inSize, int hiddenSize, int outSize){
        this.initConstants(inSize, hiddenSize, outSize);
        //init weights
        hiddenWeights = new Matrix(Maths.fillRandom(in,hidden));
        outputWeights = new Matrix(Maths.fillRandom(hidden, out));
    }
    
    public final void initConstants(int i, int h, int o){
        in = i;
        hidden = h;
        out = o;
        DhiddenWeights = new Matrix(in, hidden);
        DoutputWeights = new Matrix(hidden,out);
        for(int r = 0; r < resultsSize; r++){
            results.add(new ArrayList());
        }
        output = new double[out];
        error = new double[out];
    }
    
    public void compute(double[] inArray, double[] target){
        //takes in the input
        inputLayer = new Matrix(inArray);
                
        //calculates the hidden layer
        hiddenIn = inputLayer.mltp(hiddenWeights);
        
        //squashes the hidden results
        hiddenOut = hiddenIn.mutate("act");        
                
        //calc outputin
        outputIn = hiddenOut.mltp(outputWeights);
        outputOut = outputIn.mutate("act");
        
        //debug output
        if(debug){
            System.out.println("Weights");
            hiddenWeights.debugOutput();
            outputWeights.debugOutput();
            System.out.println("Input");
            inputLayer.debugOutput();
            System.out.println("Hidden in");
            hiddenIn.debugOutput();    
            System.out.println("Hidden act");
            hiddenOut.debugOutput();
            System.out.println("Output In");
            outputIn.debugOutput();
            System.out.println("Output act");
            outputOut.debugOutput();
            System.out.println("Compute out");
            outputOut.debugOutput();            
        }
        
        output = outputOut.self[0];
        for(int n = 0; n < out; n++){
            error[n] = 0.5*Math.pow((target[n] - output[n]),2); //error = (t-o)^2 / 2
        }
    }
    
    public void train(double[] input, double[] target, double rate, double mass){
        this.compute(input, target);
        
        for (int n = 0; n < out; n++){
            System.out.println("O: " + output[n] + " E: " + error[n]);
            double dOdnO = output[n] - target[n];
            double dnOdI = output[n] * (1 - output[n]);
            double dOdI = dOdnO * dnOdI;
            
            for(int hiddenNode = 0; hiddenNode < hidden; hiddenNode++){
                //finding delta for hidden -> output
                double dIdW = hiddenOut.get(0,hiddenNode);
                double deltaH = dOdI*dIdW;
                DoutputWeights.set(hiddenNode, n, deltaH);

                //finding delta for input -> hidden        
                double dIdHo = outputWeights.get(hiddenNode, 0);
                double dHodHi = hiddenOut.get(0, hiddenNode) * (1 - hiddenOut.get(0, hiddenNode));            
                for(int inputNode = 0; inputNode < in; inputNode++){
                    double dHidw = inputLayer.get(0, inputNode);
                    double deltaI = dOdI*dIdHo*dHodHi*dHidw;
                    DhiddenWeights.set(inputNode,hiddenNode,deltaI);
                }
            }
        }
        
        //finding momentum
        if(trained){
            int prevIndex = (resultsCount + 99) % 100;
            if(debug)System.out.println(resultsCount + " " + prevIndex);
            ArrayList prevResults = (ArrayList) results.get(prevIndex);
            Matrix prevOutput = (Matrix) prevResults.get(0);
            Matrix prevHidden = (Matrix) prevResults.get(1);
            DoutputWeights = DoutputWeights.add(prevOutput.mltp(mass));
            DhiddenWeights = DhiddenWeights.add(prevHidden.mltp(mass));
        } else{
            trained = false;
        }

        //applying learning rate
        DoutputWeights = DoutputWeights.mltp(rate);
        DhiddenWeights = DhiddenWeights.mltp(rate);
        //applying weights
        outputWeights = outputWeights.sub(DoutputWeights);
        hiddenWeights = hiddenWeights.sub(DhiddenWeights);
        
        //adding results to results array
        ArrayList singleResult = new ArrayList(){{
            add(DoutputWeights.clone());
            add(DhiddenWeights.clone());
            add(error);
        }};
        results.set(resultsCount, singleResult);
        resultsCount++;
        resultsCount = resultsCount % resultsSize;
        
    }    
}
