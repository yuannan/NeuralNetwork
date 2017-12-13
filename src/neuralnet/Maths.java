/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author blakk
 */
public class Maths {
    public static double operation(double dbIn, String operation){
        double result;
        switch(operation){
            case "double":
                result = dbIn * 2;
                break;
            case "danny":
                result = 1/dbIn;
                break;
            case "act": //the logistic function of 1/(1+ e^-x)
                result = 1/(1+ Math.exp(-dbIn));
                break;
            case "actD": // act diff
                double act = 1/(1+ Math.exp(-dbIn));
                result =  act * (1 -  act);
            default:
                result = dbIn;
                break;
        }
        
        return result;
    }
    
    public static double[][] ACopy(double[][] in){
        double[][] out = new double[in.length][in[0].length];
        
        for(int row = 0; row < in.length; row++){
            out[row] =  in[row].clone();
            //System.out.println(in[row].toString());
        }
        
        return out;
    }
    
    public static double[][] fillRandom(int rows, int cols){
        double[][] ranArray = new double[rows][cols];
                
        for(int row = 0; row < rows; row++){
            for(int col = 0; col < cols; col++){
                ranArray[row][col] = Math.random()-0.5;
                //System.out.println("row: "+ row + " col: "+ col+" value: " + ranArray[row][col]);
            }
        }
        
        return ranArray;
    }
    
    public static void writeMatrix(String outputName, Matrix mIn){
        try{
            BufferedWriter outWrite = new BufferedWriter(new FileWriter(outputName,false));
        
            for(int row = 0; row < mIn.self.length; row++){
                String currentRow = "";
                for(int col = 0; col < mIn.self[0].length; col++){
                    currentRow += mIn.self[row][col];
                    currentRow += (col == (mIn.self[0].length-1))? "": ",";
                }
                outWrite.write(currentRow);
                outWrite.newLine();
            }        
            outWrite.close();
        } catch(IOException e){
            System.out.println("Export failed!");
        }
        
    }
}
