Table data = new Table();
Table eval = new Table();
Boolean Stop = false;
double in[] = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int target = 0;
double re[][] = new double[4][16];

Network b;
void setup() {
  data = loadTable("mnist_train.csv", "header, csv");
  //eval = loadTable("mnist_test.csv", "header, csv");
  size(900, 700);

  b = new Network(data, 784, new int[]{2000}, 10);
  //b.output();
  //b.Feed(10, true);
}


void draw() {
  background(#D9D9D9);
  rect(20, 20, 50, 50, 10);
  rect(75, 20, 50, 50, 10);
  for (int i = 0; i<10; i++) {
    if (i==target) {
      fill(#D9D9D9);
    } else {
      fill(255);
    }
    rect(20+60*i, 120, 60, 30);
    rect(20+60*i, 150, 60, 30);
    fill(0);
    textSize(20);
    text("N"+str(i), 20+60*i+30, 120+textAscent());
    textSize(10);
    text((float)in[i], 20+60*i+30, 150+textAscent()*2);
  }
  for (int L = 0; L<re.length; L++) {
    for (int N = 0; N<re[L].length; N++) {
      if(L!=re.length-1){
      try{
      fill(map((float)re[L][N], -100.0, 100.0, 0.0, 255.0));
      } catch(Exception e){
      }
      } else{
      try{
      fill((float)re[L][N]*25.5);
      } catch(Exception e){
      }
      }
      ellipse(50+10*L, 200+10*N, 10, 10);
    }
  }
}

void mousePressed() {
  println(mouseX, mouseY);
  if (mouseX>20&&mouseX<20+50&&mouseY>20&&mouseY<20+50) {
    println("Training");
    thread("traint");
  }
  if (mouseX>75&&mouseX<75+50&&mouseY>20&&mouseY<20+50) {
    Stop = true;
    b.saveModel();
    println("stopping training, Saving Model");
  }
  
}

void traint() {
  b.train(0.001D, 5, true);
  b.train(0.00001D, 2000, true);
}
