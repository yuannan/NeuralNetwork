/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

/**
 *
 * @author blakk
 */

//imports
import java.util.ArrayList;
import javax.swing.table.DefaultTableModel;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;


public class Interface extends javax.swing.JFrame {
    //global vars
    double output;
    double error;
    //init global input vars
    double[] inputInputData = {};
    double inputOutputDouble;
    double rate;
    double mass;
    String messageBoxString;
    //init global training sets
    ArrayList<double[]> trainingInput = new ArrayList();
    ArrayList<Double> trainingOutput = new ArrayList();
    //init table
    Object[][] logicTableData = {
        {"","",""},
        {"","",""},
        {"","",""},
        {"","",""}
            
    };
    String[] logicTableHeader = {"A","B","Out"};
    DefaultTableModel logicmodel = new DefaultTableModel(logicTableData, logicTableHeader);
    
    //init new network
    Network net = new Network(2,3,1);
    
    public Interface() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        gateCombo = new javax.swing.JComboBox<>();
        gateLabel = new javax.swing.JLabel();
        generateButton = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        logicTable = new javax.swing.JTable();
        inputLabel = new javax.swing.JLabel();
        input0 = new javax.swing.JTextField();
        input1 = new javax.swing.JTextField();
        outputLabel = new javax.swing.JLabel();
        outputBox = new javax.swing.JTextField();
        errorLabel = new javax.swing.JLabel();
        errorBox = new javax.swing.JTextField();
        calcButton = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        trainingOutputInput = new javax.swing.JTextField();
        newNetworkButton = new javax.swing.JButton();
        trainButton = new javax.swing.JButton();
        outputInputLabel = new javax.swing.JLabel();
        messageLabel = new javax.swing.JLabel();
        jScrollPane2 = new javax.swing.JScrollPane();
        messageBox = new javax.swing.JTextArea();
        dumpButton = new javax.swing.JButton();
        autodataCheckbox = new javax.swing.JCheckBox();
        roundsInput = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        rateInputLabel = new javax.swing.JLabel();
        rateInput = new javax.swing.JTextField();
        jLabel3 = new javax.swing.JLabel();
        massInput = new javax.swing.JTextField();
        outputFileButton = new javax.swing.JButton();
        hiddenFileButton = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        gateCombo.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "AND", "OR", "NAND", "XOR" }));

        gateLabel.setText("Gate");

        generateButton.setText("Generate");
        generateButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                generateButtonActionPerformed(evt);
            }
        });

        logicTable.setModel(logicmodel);
        jScrollPane1.setViewportView(logicTable);

        inputLabel.setText("Input");

        input0.setText("0");

        input1.setText("0");

        outputLabel.setText("Output");

        outputBox.setText("output");

        errorLabel.setText("Error");

        errorBox.setText("error");

        calcButton.setText("Calculate");
        calcButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                calcButtonActionPerformed(evt);
            }
        });

        jLabel1.setText("Training");

        trainingOutputInput.setText("0");

        newNetworkButton.setText("NEW NETWORK");
        newNetworkButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                newNetworkButtonActionPerformed(evt);
            }
        });

        trainButton.setText("Train");
        trainButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                trainButtonActionPerformed(evt);
            }
        });

        outputInputLabel.setText("Output");

        messageLabel.setText("Message: ");

        messageBox.setColumns(20);
        messageBox.setRows(5);
        jScrollPane2.setViewportView(messageBox);

        dumpButton.setText("DUMP Weights");
        dumpButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                dumpButtonActionPerformed(evt);
            }
        });

        autodataCheckbox.setText("autoData");

        roundsInput.setText("1");

        jLabel2.setText("TRAINING INPUTS");

        rateInputLabel.setText("Rate");

        rateInput.setText("1");
        rateInput.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                rateInputActionPerformed(evt);
            }
        });

        jLabel3.setText("Mass");

        massInput.setText("0.3");

        outputFileButton.setText("Output Path");
        outputFileButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                outputFileButtonActionPerformed(evt);
            }
        });

        hiddenFileButton.setText("Hidden Path");
        hiddenFileButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hiddenFileButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(35, 35, 35)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(messageLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jScrollPane2)
                        .addContainerGap())
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addGap(5, 5, 5)
                                        .addComponent(gateLabel)
                                        .addGap(18, 18, 18)
                                        .addComponent(gateCombo, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                            .addGroup(layout.createSequentialGroup()
                                                .addComponent(inputLabel)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                                .addComponent(input0, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE))
                                            .addComponent(jLabel2, javax.swing.GroupLayout.Alignment.LEADING))
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(input1, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(calcButton)
                                    .addComponent(generateButton)))
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(outputInputLabel)
                                    .addComponent(rateInputLabel)
                                    .addComponent(jLabel3))
                                .addGap(18, 18, 18)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(trainingOutputInput, javax.swing.GroupLayout.DEFAULT_SIZE, 50, Short.MAX_VALUE)
                                    .addComponent(rateInput)
                                    .addComponent(massInput))
                                .addGap(229, 229, 229)
                                .addComponent(trainButton)
                                .addGap(0, 41, Short.MAX_VALUE))
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(outputLabel)
                                    .addComponent(errorLabel))
                                .addGap(18, 18, 18)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(outputBox, javax.swing.GroupLayout.PREFERRED_SIZE, 224, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(errorBox, javax.swing.GroupLayout.PREFERRED_SIZE, 224, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(dumpButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(outputFileButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)))
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 300, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jLabel1)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(autodataCheckbox)
                                .addGap(18, 18, 18)
                                .addComponent(roundsInput, javax.swing.GroupLayout.PREFERRED_SIZE, 89, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addComponent(hiddenFileButton, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(newNetworkButton, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(generateButton)
                            .addComponent(gateLabel)
                            .addComponent(gateCombo, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jLabel2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(inputLabel)
                            .addComponent(input0, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(input1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(calcButton))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(outputInputLabel)
                            .addComponent(trainingOutputInput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE))
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(6, 6, 6)
                                .addComponent(rateInput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(10, 10, 10)
                                .addComponent(rateInputLabel)))
                        .addGap(3, 3, 3)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jLabel3)
                            .addComponent(massInput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(outputLabel)
                            .addComponent(outputBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(dumpButton)
                            .addComponent(newNetworkButton)))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(autodataCheckbox)
                            .addComponent(jLabel1)
                            .addComponent(trainButton)
                            .addComponent(roundsInput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(errorLabel)
                    .addComponent(errorBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(outputFileButton)
                        .addComponent(hiddenFileButton)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 285, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(messageLabel)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void generateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_generateButtonActionPerformed
        //get and clear arrays
        int gateType = gateCombo.getSelectedIndex();
        trainingInput.clear();
        trainingOutput.clear();
        
        int row = 0;
        //Start generation for gates
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                double outputBol = 0;
                switch(gateType){
                    case 0:
                        logicTableHeader[2]="AND";
                        outputBol = i & j;
                        break;
                    case 1:
                        logicTableHeader[2]="OR";
                        outputBol = i | j;
                        break;
                    case 2:
                        logicTableHeader[2]="NAND";
                        outputBol = ((i & j) == 1)? 0 : 1;
                        break;
                    case 3:
                        logicTableHeader[2]="XOR";
                        outputBol = i ^ j;
                        break;
                    case -1:
                        System.out.println("Bad gate index -1");
                        break;
                    default:
                        System.out.println("Error in gate selection");
                        break;
                }
                //update table
                logicmodel.setValueAt(i, row, 0);
                logicmodel.setValueAt(j, row, 1);
                logicmodel.setValueAt((int) outputBol, row, 2);
                row++;
                //System.out.println(outputBol);
                //update arrays
                trainingInput.add(new double[] {i,j});                              
                trainingOutput.add(outputBol);
            }
        }
    }//GEN-LAST:event_generateButtonActionPerformed

    private void newNetworkButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_newNetworkButtonActionPerformed
        // TODO add your handling code here:
        net = new Network(2,3,1);
    }//GEN-LAST:event_newNetworkButtonActionPerformed

    private void calcButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_calcButtonActionPerformed
        // TODO add your handling code here:
        updateInputs();
        net.compute(inputInputData, inputOutputDouble);
        updateOutputs();
    }//GEN-LAST:event_calcButtonActionPerformed

    private void trainButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_trainButtonActionPerformed
        updateInputs();
        for(int rounds = 0; rounds < Integer.valueOf(roundsInput.getText()); rounds++){
            if(autodataCheckbox.isEnabled()){
                //updates the training set
                this.generateButtonActionPerformed(evt);
                //trains the network on one set
                for(int set = 0; set < trainingInput.size(); set++ ){
                    net.train(trainingInput.get(set), trainingOutput.get(set), rate, mass);
                }
            } else{
                net.train(inputInputData, inputOutputDouble, rate, mass);
            }
        }
        updateOutputs();        
    }//GEN-LAST:event_trainButtonActionPerformed

    private void dumpButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_dumpButtonActionPerformed
        // TODO add your handling code here:
        String hiddenWeightsString = net.hiddenWeights.debugReturn();
        String outputWeightsString = net.outputWeights.debugReturn();
        messageBox.setText(hiddenWeightsString+outputWeightsString);
        //init current time
        DateFormat df = new SimpleDateFormat("yyyyMMddHHmmss");
        Date dateobj = new Date();
        String timestamp = df.format(dateobj);
        //init file writer
        try{
            BufferedWriter bwH = new BufferedWriter(new FileWriter("weights/hidden"+timestamp));
            BufferedWriter bwO = new BufferedWriter(new FileWriter("weights/output"+timestamp));
            bwH.write(hiddenWeightsString);
            bwO.write(outputWeightsString);
            bwH.close();
            bwO.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }//GEN-LAST:event_dumpButtonActionPerformed

    private void rateInputActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_rateInputActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_rateInputActionPerformed

    private void outputFileButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_outputFileButtonActionPerformed
        //updates outputWeights with the new file
        updateInputs();
        net.outputWeights = new Matrix(messageBoxString);
    }//GEN-LAST:event_outputFileButtonActionPerformed

    private void hiddenFileButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hiddenFileButtonActionPerformed
        //updates hiddenWeights with the new file
        updateInputs();
        net.hiddenWeights = new Matrix(messageBoxString);
    }//GEN-LAST:event_hiddenFileButtonActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Interface().setVisible(true);
            }
        });
    }
    
    public void updateInputs(){
        inputInputData = new double[] {Double.valueOf(input0.getText()), Double.valueOf(input1.getText())};
        inputOutputDouble = Double.valueOf(trainingOutputInput.getText());
        rate = Double.valueOf(rateInput.getText());
        mass = Double.valueOf(massInput.getText());
        messageBoxString = messageBox.getText();
    }
    
    public void updateOutputs(){
        output = net.output;
        error = net.error;
        outputBox.setText(String.valueOf(output));
        errorBox.setText(String.valueOf(error));
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JCheckBox autodataCheckbox;
    private javax.swing.JButton calcButton;
    private javax.swing.JButton dumpButton;
    private javax.swing.JTextField errorBox;
    private javax.swing.JLabel errorLabel;
    private javax.swing.JComboBox<String> gateCombo;
    private javax.swing.JLabel gateLabel;
    private javax.swing.JButton generateButton;
    private javax.swing.JButton hiddenFileButton;
    private javax.swing.JTextField input0;
    private javax.swing.JTextField input1;
    private javax.swing.JLabel inputLabel;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JTable logicTable;
    private javax.swing.JTextField massInput;
    private javax.swing.JTextArea messageBox;
    private javax.swing.JLabel messageLabel;
    private javax.swing.JButton newNetworkButton;
    private javax.swing.JTextField outputBox;
    private javax.swing.JButton outputFileButton;
    private javax.swing.JLabel outputInputLabel;
    private javax.swing.JLabel outputLabel;
    private javax.swing.JTextField rateInput;
    private javax.swing.JLabel rateInputLabel;
    private javax.swing.JTextField roundsInput;
    private javax.swing.JButton trainButton;
    private javax.swing.JTextField trainingOutputInput;
    // End of variables declaration//GEN-END:variables
}
