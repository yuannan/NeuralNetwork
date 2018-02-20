package NeuralNet;
import java.util.ArrayList;

public class Network {
    //settings up tracker
    int resultsSize = 100;
    int resultsCount = 0;
    int memory = 10;
    ArrayList<ArrayList<Matrix>> results = new ArrayList(resultsSize);
    boolean debug = false;
    boolean trained = false;
    int trainingCount = 0;
    
    //init constants
    Matrix hiddenWeights, outputWeights; //weights
    Matrix DhiddenWeights, DoutputWeights; //delta weights
    Matrix inputIn, inputOut, hiddenIn, hiddenOut, outputIn, outputOut; //layer values
    Matrix bInput, bHidden, bOutput; // bias values
    int inputNodes, hiddenNodes, outputNodes;
    
    public double[] output;
    public double[] error;
    
    //init of a 3 layer network with weights and bias
    public Network(int inSize, int hiddenSize, int outSize, Matrix inputToHidden, Matrix hiddenToOut, Matrix inputBias, Matrix hiddenBias, Matrix outputBias){
        //setting network constants
        this.initConstants(inSize, hiddenSize, outSize);
        //init weights
        hiddenWeights = inputToHidden;
        outputWeights = hiddenToOut;
        //init bias
        bInput = inputBias;
        bHidden = hiddenBias;
        bOutput = outputBias;
        
        //validating network weight sizes
        if(hiddenWeights.rows != inputNodes || hiddenWeights.cols != hiddenNodes) System.out.println("Inproper hidden weight size!");
        if(outputWeights.rows != hiddenNodes  || outputWeights.cols != outputNodes) System.out.println("Inproper output weight size!");
        //validating network bias sizes
        if(bInput.rows != 1 || bInput.cols != inputNodes) System.out.println("Inproper input bias size!");
        if(bHidden.rows !=1 || bHidden.cols != hiddenNodes) System.out.println("Inproper hidden bias size!");
        if(bOutput.rows != 1 || bOutput.cols != outputNodes) System.out.println("Inproper output bias size!");
    
    }
    
    //init of a 3 layer network with random weights
    public Network(int inSize, int hiddenSize, int outSize){
        this.initConstants(inSize, hiddenSize, outSize);
        //init weights
        hiddenWeights = new Matrix(Maths.fillRandom2DArray(inputNodes,hiddenNodes));
        outputWeights = new Matrix(Maths.fillRandom2DArray(hiddenNodes, outputNodes));
        //init bias
        bInput = new Matrix(Maths.fillRandom2DArray(1, inputNodes));
        bHidden = new Matrix(Maths.fillRandom2DArray(1, hiddenNodes));
        bOutput = new Matrix(Maths.fillRandom2DArray(1, outputNodes));
    }
    
    public final void initConstants(int i, int h, int o){
        //init constant sizes
        inputNodes = i;
        hiddenNodes = h;
        outputNodes = o;
        //delta weights
        DhiddenWeights = new Matrix(inputNodes, hiddenNodes);
        DoutputWeights = new Matrix(hiddenNodes,outputNodes);
        //init result array
        for(int r = 0; r < resultsSize; r++){
            results.add(new ArrayList());
        }
        //init output arrays
        output = new double[outputNodes];
        error = new double[outputNodes];
    }
    
    public void compute(double[] inArray, double[] target){
        //input
        inputIn = new Matrix(inArray);
        inputOut = (inputIn.add(bInput)).mutate("act");
                
        //hidden
        hiddenIn = inputOut.mltp(hiddenWeights);
        hiddenOut = (hiddenIn.add(bHidden)).mutate("act");
                
        //output
        outputIn = hiddenOut.mltp(outputWeights);
        outputOut = (outputIn.add(bOutput)).mutate("act");
        
        //debug outputs
        if(debug){
            System.out.println("Hidden and Output Weights");
            hiddenWeights.debugOutput();
            outputWeights.debugOutput();
            System.out.println("Input, Hidden and Output bias");
            bInput.debugOutput();
            bHidden.debugOutput();
            bOutput.debugOutput();
            System.out.println("Input in");
            inputIn.debugOutput();
            System.out.println("Input out");
            inputOut.debugOutput();
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
        for(int n = 0; n < outputNodes; n++){
            error[n] = 0.5*Math.pow((target[n] - output[n]),2); //error = (t-o)^2 / 2
        }
    }
    
    public void train(double[] input, double[] target, double rate, double biasRate, double mass){
        this.compute(input, target);
        
        for (int outputNode = 0; outputNode < outputNodes; outputNode++){
            if(debug)System.out.println("O: " + output[outputNode] + " E: " + error[outputNode]);
            double dOdoO = output[outputNode] - target[outputNode];
            double dnOdoI = output[outputNode] * (1 - output[outputNode]);
            double dOdoI = dOdoO * dnOdoI;
            //updating bias for output
            bOutput.inc(0,outputNode,-(dOdoI*biasRate));
            
            //finding delta for each hidden -> output node
            for(int hiddenNode = 0; hiddenNode < hiddenNodes; hiddenNode++){
                double doIdhW = hiddenOut.get(0,hiddenNode);
                double dOdhW = dOdoI*doIdhW;
                DoutputWeights.set(hiddenNode, outputNode, dOdhW);
                
                double doIdhO = outputWeights.get(hiddenNode, 0);
                double dhOdhI = hiddenOut.get(0, hiddenNode) * (1 - hiddenOut.get(0, hiddenNode));
                
                double dOdhI = dOdoI*doIdhO*dhOdhI;                
                bHidden.inc(0,hiddenNode,-(dOdhI*biasRate)); //hidden bias
                        
                //finding delta for each input -> hidden node
                for(int inputNode = 0; inputNode < inputNodes; inputNode++){
                    double dhIdiW = inputOut.get(0, inputNode);
                    double dOdiW = dOdhI * dhIdiW;
                    DhiddenWeights.set(inputNode,hiddenNode,dOdiW);
                    
                    double dhIdiO = hiddenWeights.get(inputNode, 0);
                    double diOdiI = inputOut.get(0,inputNode) * (1 - inputOut.get(0,inputNode));
                    
                    double dOdiI = dOdhI * dhIdiO * diOdiI;
                    bInput.inc(0,inputNode,-(dOdiI*biasRate));              //updating the bias for inputs
                }
            }
        }
        
        //finding momentum
        if(trained){
            int setIndex = 0;
            Matrix momHidden = new Matrix(inputNodes,hiddenNodes);
            Matrix momOutput = new Matrix(hiddenNodes,outputNodes);
            for(int set = 0; set < memory; set++){
                setIndex = (resultsCount + 99 - set) % 100;
                if(debug)System.out.println(resultsCount + " " + setIndex);
                momHidden.add(results.get(setIndex).get(0));
                momOutput.add(results.get(setIndex).get(1));
            }
            
            momHidden = momHidden.mutate("div", memory);
            momOutput = momOutput.mutate("div", memory);
            DhiddenWeights = DhiddenWeights.add(momHidden.mutate("mltp",mass));
            DoutputWeights = DoutputWeights.add(momOutput.mutate("mltp",mass));
        } else{
            trained = false;
        }

        //applying learning rate
        DoutputWeights = DoutputWeights.mutate("mltp",rate);
        DhiddenWeights = DhiddenWeights.mutate("mltp",rate);
        //applying weights
        outputWeights = outputWeights.sub(DoutputWeights);
        hiddenWeights = hiddenWeights.sub(DhiddenWeights);
        
        //adding results to results array
        ArrayList singleResult = new ArrayList(){{
            add(DhiddenWeights.clone());
            add(DoutputWeights.clone());
            add(error);
        }};
        results.set(resultsCount, singleResult);
        resultsCount++;
        resultsCount = resultsCount % resultsSize;
        
        //increment samples trained
        trainingCount++;
    }
    
    public int getTrainingCount(){
        return trainingCount;
    }
}