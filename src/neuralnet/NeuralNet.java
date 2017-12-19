package neuralnet;

/**
 *
 * @author blakk
 */

import java.util.Scanner;
import java.util.ArrayList;


public class NeuralNet {
    static String version = "1.1.0";
    
    public static void main(String[] args) {
        Interface face = new Interface();
        face.setTitle("Neural Network 9001 V"+version);
        face.setVisible(true);
    }
}
