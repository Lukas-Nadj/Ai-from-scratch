class ReLu {

  // ------data--------
  Table data;
  int layers = 0;
  int outputs = 0;
  //-------weights and bias
  double[][][] weights;
  double[][] bias;
  double[] err = new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};


  // ------initialisation--------
  ReLu(Table data, int inputs, int[] layers, int output) {
    this.data = data;
    this.layers = layers.length+1;
    this.outputs = output;

    bias = new double[layers.length+1][];
    weights = new double[layers.length+1][][];

    weights[0] = new double[layers[0]][inputs];
    //bias[0] = new double[layers[0]];
    for (int i = 1; i<layers.length; i++) {
      //layers   =             neurons    weights
      weights[i] = new double[layers[i]][layers[i-1]];
    }

    for (int i = 0; i<layers.length; i++) {
      bias[i] = new double[layers[i]];
    }
    bias[layers.length] = new double[outputs];
    weights[layers.length] = new double[outputs][layers[layers.length-1]];



    for (int L = 0; L<weights.length/*LAYERS*/; L++) {
      for (int N = 0; N<weights[L].length/*NEURONS*/; N++) {
        for (int W = 0; W<weights[L][N].length/*NEURONS*/; W++) {
          weights[L][N][W] = (double)random(0.2);
        }
      }
    }
    for (int L = 0; L<weights.length/*LAYERS*/; L++) {
      for (int N = 0; N<weights[L].length/*NEURONS*/; N++) {
        bias[L][N] = (double)random(-0.0001, 0.00001);
      }
    }
  }

  void Feed(int input, Boolean output) {
    TableRow a = data.getRow(input);
    double[] inputs = new double[784];
    double[][] results = new double[layers][0];
    for (int i = 1; i<785; i++) {
      inputs[i-1] = (double)map(a.getInt(i), 0, 255, 0.0, 1.0);
    }
    double[] targets = new double[outputs];
    for (int j = 0; j < outputs; j++) {
      targets[j] = j == a.getInt(0) ? (double)1D : (double)0D;
    }

    //iterate each layer
    double[][] activations = new double[weights.length][];
    for (int L = 0; L<weights.length/*LAYERS*/; L++) {
      activations[L] = new double[weights[L].length];
      //iterate over each neuron
      for (int N = 0; N<weights[L].length/*NEURONS*/; N++) {
        double[] temp = Activation_funtion(inputs, weights[L][N], bias[L][N]);
        activations[L][N] = temp[0]; //activation
      }
      inputs = activations[L];
    }
    re = activations;

    double sum = 0;
    in = inputs;
    target = a.getInt(0);
    if (output) {
      println("      label", a.getInt(0));
      for (int i = 0; i<outputs; i++) {
        if (i==a.getInt(0)) {
          print(">>");
        } else {
          print("!=");
        }
        println(i, "   ", (float)(re[weights.length-1][i]), "   ", (float)err[i]);
      }
      print("\n");
      for (int x = 0; x<results.length; x++) {
        for (int y = 0; y<results[x].length; y++) {
          //print(results[x][y]);
          sum += results[x][y];
        }
        println(sum);
      }
    }
  }

  double[] Activation_funtion(double[] inputs, double[] weight, double bias) {
    double sum = 0.0;
    for (int i = 0; i < weight.length; i++) {
      sum += inputs[i] * weight[i];
    }
    if (sum<0) {
      return new double[]{0.0, sum};
    }
    return new double[]{sum+bias, sum};
  }

  double gradient(double x) {
    return x>0 ? (double)1.0 : (double)0.00000000000000001;
  }
  void saveModel() {
    /*for (int l = 0; l<layers; l++) {
     String[] data =
     }*/
  }

  void train(double learningRate, int epoch, Boolean output) {
    for (int e = 0; e<epoch; e++) {
      if (output) {
        println("Epoch", e);
      }

      this.Feed((int)random(data.getRowCount()-1), output);
      int mi = millis();
      for (int d = 0; d<data.getRowCount(); d++) {
        if (Stop) {
          e = epoch;
          Stop = false;
          return;
        }
        if (output&&millis()>mi+2000) {
          mi = millis();
          println("Epoch", e);
          this.Feed((int)random(data.getRowCount()-1), output);
        }

        //-------------------FEEEED---------------------
        TableRow a = data.getRow(d);  //grab data
        double[] inputs = new double[784]; //create array for it
        double[] inputLayer = new double[784];
        double[][] activations = new double[layers][0];  //Activations
        double[][] weighted_sum = new double[layers][0];  //Weighted sum

        for (int i = 1; i<785; i++) {
          inputs[i-1] = (double)map(a.getInt(i), 0, 255, 0.0, 1.0); //map and load into array
        }
        arrayCopy(inputs, inputLayer);
        
        double[] targets = new double[outputs];
        for (int j = 0; j < outputs; j++) {  //targets, finding y
          targets[j] = j == a.getInt(0) ? (double)1.0D : (double)0.0D;
        }

        //iterate each layer
        for (int L = 0; L<weights.length/*LAYERS*/; L++) {
          activations[L] = new double[weights[L].length];
          weighted_sum[L] = new double[weights[L].length];
          //iterate over each neuron
          for (int N = 0; N<weights[L].length/*NEURONS*/; N++) {
            double[] temp = Activation_funtion(inputs, weights[L][N], bias[L][N]);
            activations[L][N] = temp[0]; //activation
            weighted_sum[L][N] =temp[1]; //weighted sum
          }
          inputs = activations[L];
        }
        re = activations;
        //printArray(activations[0]);


        //---------------------BACKPROPPP----------------------------


        //initialize error array

        double[] errors = new double[targets.length];
        double error = 0.0;
        for (int g = 0; g<outputs/*NEURONS*/; g++) {
          errors[g] = java.lang.Math.pow(targets[g]-activations[weights.length-1][g], 2);    //MSE
          error += java.lang.Math.pow(targets[g]-activations[weights.length-1][g], 2);
        }
        err = errors;



        double preverror = 0.0;
        for (int L = weights.length-1; L>0/*NEURONS*/; L--) {  //we aren't doing the last layer, because it's activations are the input layer

          for (int N = 0; N<weights[L].length; N++) {
            double layer_error = activations[L-1][N]*gradient(weighted_sum[L][N])*(double)(error);
            bias[L][N] -= layer_error*learningRate;
            for (int W = 0; W<weights[L][N].length; W++) {
              double weights_gradient = activations[L-1][W]*layer_error;
              weights[L][N][W] -= weights_gradient*learningRate;
              preverror += weights[L][N][W]*weighted_sum[L][N]*layer_error;
            }
          }
          error = preverror;
        }

        for (int N = 0; N<weights[0].length; N++) {
          double layer_error = gradient(weighted_sum[0][N])*(double)(error);
          bias[0][N] -= layer_error*learningRate;
          for (int W = 0; W<weights[0][N].length; W++) {
            double weights_gradient = inputLayer[W]*layer_error;
            weights[0][N][W] -= weights_gradient*learningRate;
          }
        }
      } //iterate dataset end
    } // epochs end
  } //function end.





  void output() {
    for (int i = 0; i<weights.length-1; i++) { //iterate layer
      print("layer", i, "    ");
      for (int a = 0; a<weights[i].length; a++) { //iterate over neurons
        print(weights[i][a].length, "   ");
      }
      print("\n");
    }
    print("output:  ");
    for (int a = 0; a<weights[weights.length-1].length; a++) { //iterate over output layer
      print(weights[weights.length-1][a].length, "  ");
    }
  }




  // ------Training--------
  /*
  void train(double learningRate, int epochs) {
   for (int e = 0; e<epochs; e++) {
   for (int i = 1; i < data.getRowCount(); i++) {
   TableRow row = data.getRow(i);
   
   }
   }
   }
   */
}
