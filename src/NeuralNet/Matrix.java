package NeuralNet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class Matrix {
    public double[][] self;
    public int rows;
    public int cols;
    
    // <editor-fold desc="init methods">
    
    public Matrix(int row, int col){
        this.self = new double[row][col];
        rows = row;
        cols = col;
    }
    
    //inits a new matrix on based on the in array
    public Matrix(double[][] in){
        self = Maths.ACopy(in);
        rows = self.length;
        cols = self[0].length;
    }
    
    //inits a one row Matrix
    public Matrix(double[] in){
        this.self = new double[1][in.length];
        self[0] = in.clone();
        rows = self.length;
        cols = self[0].length;
    }
        
    //inits a matrix from a filename
    public Matrix(String filename){
        try{
            //inits tools
            BufferedReader br = new BufferedReader(new FileReader(filename));
            ArrayList<String[]> stringStore = new ArrayList<String[]>();
            
            //imports strings to an temp ArrayList
            String in = "";
            while((in = br.readLine()) != null){
                String[] currentLine = in.split(",");
                stringStore.add(currentLine);
            }
            
            //creates self (matrix data) from the temp store
            this.rows = stringStore.size();
            this.cols = stringStore.get(0).length;
            self = new double[rows][cols];
            for(int row = 0; row < rows; row++){
                for(int col = 0; col < cols; col++){
                    self[row][col] = Double.valueOf(stringStore.get(row)[col]);                    
                }
            }
        } catch (Exception e){
            System.out.println("IO Error");
        }
    }
    
    //</editor-fold>
    
    //<editor-fold desc="get and sets">
        
    //get whole matrix in array form
    public double[][] getArray(){
        return self;
    }
    
    //get singular item
    public double get(int row, int col){
        return self[row][col];
    }
    
    //set whole matrix
    public void set(double[][] in){
        self = in.clone();
    }
    
    //set singular item
    public void set(int row, int col, double in){
        self[row][col] = in;
    }
    
    //increment a singular item
    public void inc(int row, int col, double in){
        self[row][col] += in;
    }
    
    //sets a row of the matrix
    public void setRow(int row, double[] in){
        if(in.length == cols){
            self[row] = in.clone();
        }else{
            System.out.println("setRow size does not match!");
            System.out.println("in: "+ in.length);
            System.out.println("rows: "+ rows);   
        }
    }
    
    public void setCol(int col, double[] in){
        if(in.length == rows){
            for(int row = 0; row < rows; row++){
                self[row][col] = in[row];
            }
        }else{
            System.out.println("setCol size does not match!");
            System.out.println("in: "+ in.length);
            System.out.println("cols: "+ cols);
        }
    }
    
//</editor-fold>
    
    //<editor-fold desc="maths methods">
    
    //adds 2 matricies together -- Tested working
    public Matrix add(Matrix aIn){
        Matrix aRes = new Matrix(self);
        if(aIn.rows == rows && aIn.cols == cols){
            for(int row = 0; row < self.length; row++){
                for(int col = 0; col < self[0].length; col++){
                    aRes.set(row, col, aRes.get(row, col) + aIn.self[row][col]);
                }                
            }
        }
        return aRes;
    }
    
    //takes 2 matricies from each other -- Tested working
    public Matrix sub(Matrix aIn){
        Matrix aRes = new Matrix(self);
        if(aIn.rows == rows && aIn.cols == cols){
            for(int row = 0; row < self.length; row++){
                for(int col = 0; col < self[0].length; col++){
                    aRes.set(row, col, aRes.get(row, col) - aIn.self[row][col]);
                }                
            }
        }
        return aRes;
    }
    
    //multiply matrix. self = A, in = B returns A.B
    public Matrix mltp(Matrix mIn){
        int size = -1;
        if(cols != mIn.rows){
            System.out.println("Illegal multiply");
            System.out.println("P("+rows+","+cols+")");
            System.out.println("I("+mIn.rows+","+mIn.cols+")");
        } else{
            size = cols;
        }
        
        double sum;
        Matrix mRes = new Matrix(rows, mIn.cols);
        for(int row = 0; row < this.rows; row++){
            //System.out.println("R:"+row);
            for(int col = 0; col < mIn.cols; col++){
                //System.out.println("C:"+col);
                sum = 0;
                for(int index = 0; index < size; index++){
                    //System.out.println("I:"+index);
                    sum += self[row][index]*mIn.self[index][col];
                }
                mRes.set(row, col, sum);
            }
        }
        return mRes;
        
    }
    
    //new wrapper for mutate function for legacy support
    public Matrix mutate(String in, double modIn){
        return this.mutateToMaths(in, modIn);
    }
    
    public Matrix mutate(String in){
        return this.mutateToMaths(in, 0);
    }
    
    public Matrix mutateToMaths(String in, double modIn){
        Matrix mRes = new Matrix(self);
        for(int row = 0; row < mRes.self.length; row++){
            for(int col = 0; col < mRes.self[0].length; col++){
                mRes.set(row, col, Maths.operation(in, mRes.self[row][col], modIn));
            }
        }
        
        return mRes;
    }
    
    public Matrix trans(){
        Matrix in = new Matrix(self);
        Matrix trans = new Matrix(in.cols, in.rows);
        
        for(int inR = 0; inR < in.rows; inR++){
            trans.setCol(inR, in.self[inR]);
        }
        
        return trans;
    }
    
    //</editor-fold>
    
    //debug output -- Tested working
    public void debugOutput(){
        System.out.println(self.length+"x"+self[0].length+" matrix");
        
        for(int row = 0; row < self.length; row++){
            String currentRow = "";
            for(int col = 0; col < self[0].length; col++){
                //System.out.println("R: "+ row +" C: "+ col + " V: " + self[row][col]);
                currentRow += self[row][col] + " ";
            }
            System.out.println(currentRow);
        }
        System.out.println();
        
    }
    
    public String debugReturn(){
        System.out.println(self.length+"x"+self[0].length+" matrix");
        String output = "";
        for(int row = 0; row < self.length; row++){
            String currentRow = "";
            for(int col = 0; col < self[0].length; col++){
                //System.out.println("R: "+ row +" C: "+ col + " V: " + self[row][col]);
                currentRow += self[row][col];
                currentRow += (col == self[0].length - 1)? "" : ",";
            }
            output+=(currentRow+"\n");
        }
        return output;
    }
    
    @Override
    public Matrix clone(){
        return new Matrix(self);
    }
    
}
