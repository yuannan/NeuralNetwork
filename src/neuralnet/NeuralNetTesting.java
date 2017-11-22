package neuralnet;

/**
 *
 * @author blakk
 */

import java.util.Arrays;

public class NeuralNetTesting {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        // TODO code application logic here
        double[][] a = {{1,3,5},
                        {2,4,6}};
        double[][] b = {{7,8},
                        {9,10},
                        {11,12}};
        
        Matrix mA = new Matrix(a);
        //Matrix mB = new Matrix(b);
        //mA.debugOutput();
        //mB.debugOutput();
        Matrix mC = mA.trans();
        Matrix mB = mA.mltp(mC);
        
        
        mA.debugOutput();
        mC.debugOutput();
        mB.debugOutput();
    }    
}
