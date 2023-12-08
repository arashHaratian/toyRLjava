package org.RL.SimpleREINFORCE;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.apache.commons.collections4.map.MultiKeyMap;


public class SimpleREINFORCE {

    public static double[] softmax(double[] values) {
        final double sum = Arrays.stream(values).map(Math::exp).sum();
        return DoubleStream.of(values).map(x -> Math.exp(x) / sum).toArray();
    }

    public static int sampleElement(int[] elements, double[] probabilities) {
        Random random = new Random();
        double p = random.nextDouble();
        double cumulativeProbability = 0.0;

        for (int i = 0; i < elements.length; i++) {
            cumulativeProbability += probabilities[i];
            if (p <= cumulativeProbability) {
                return elements[i];
            }
        }
        // Fallback in case of rounding errors
        return elements[elements.length - 1];
    }

    public static double[] multiplyVectorByMatrix(double[] theta, double[][] stateFeatures) {
        int rows = stateFeatures.length;
        int cols = stateFeatures[0].length;
        double[] result = new double[cols];

        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                result[j] += theta[i] * stateFeatures[i][j];
            }
        }

        return result;
    }



    MultiKeyMap<Object, double[]> stateFeatures;
    private double[] theta;

    private ArrayList<Double> rewards;
    private ArrayList<Integer> states, actions;
    private double gamma;
    private int obsShape, actionShape;
    private double lr;


    public SimpleREINFORCE(int numObs, int numActions, double lr, double gamma){
        obsShape = numObs;
        actionShape = numActions;
        this.gamma = gamma;
        this.lr = lr;
        this.stateFeatures= new MultiKeyMap<>();

        stateFeatures.put(1, 0, new double[]{1,0,0,0,0,0});
        stateFeatures.put(1, 1, new double[]{0,1,0,0,0,0});
        stateFeatures.put(2, 0, new double[]{0,0,1,0,0,0});
        stateFeatures.put(2, 1, new double[]{0,0,0,1,0,0});
        stateFeatures.put(3, 0, new double[]{0,0,0,0,1,0});
        stateFeatures.put(3, 1, new double[]{0,0,0,0,0,1});


        Random random = new Random();
        this.theta = random.doubles(numObs*2).toArray();
        this.states = new ArrayList<>();
        this.actions = new ArrayList<>();
        this.rewards = new ArrayList<>();


    }

    public SimpleREINFORCE(int numObs, int numActions){
        this(numObs, numActions, 0.1, 0.6);
    }

    public double[] pi(int state){

        double[][] currentStateFeature = new double[this.actionShape][this.actionShape * this.obsShape];
        for(int i = 0; i < this.actionShape; i++){
            currentStateFeature[i] = this.stateFeatures.get(state, i);
        }
        double[] hValue = multiplyVectorByMatrix(this.theta, currentStateFeature);
        double[] probs = softmax(hValue);

        return probs;
    }

    public int chooseAction(int state){
        return sampleElement(new int[]{0, 1}, pi(state));
    }

    private double[] calculateReturns(){
        double[] movingReturn = new double[this.rewards.size()];
        double episodeReturn = 0;
        for (int i = this.rewards.size() - 1; i >= 0; i--) {
            episodeReturn = this.rewards.get(i) + this.gamma * episodeReturn;
            movingReturn[i] = episodeReturn;
        }
        return movingReturn;
    }

    private double[] calculateGrad(int state, int action, double[] probs){

        double[] featureVector = this.stateFeatures.get(state, action);

        double[][] currentStateFeatures = new double[this.actionShape][this.actionShape * this.obsShape];
        for(int i = 0; i < this.actionShape; i++){
            currentStateFeatures[i] = this.stateFeatures.get(state, i);
        }

        double[] res = multiplyVectorByMatrix(probs, currentStateFeatures);

        return IntStream.range(0, res.length).mapToDouble(i -> featureVector[i] - res[i]).toArray();

    }
    
    public void update(){
        double[] movingReturn = calculateReturns();
        int state, action;

//        double discountPower = 1;
        for(int i = 0; i < movingReturn.length; i++){
            final int tempi = i;
            state = states.get(i);
            action = actions.get(i);
            double[] probs = pi(state);
            double[] grad = calculateGrad(state, action, probs);

            theta = IntStream.range(0, theta.length).mapToDouble(j -> theta[j] + this.lr * movingReturn[tempi] * grad[j]).toArray();
        }

        states.clear();
        rewards.clear();
        actions.clear();
    }

    public void save(){
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("./saved_agent/thetaREINFORCE.txt"))) {
            out.writeObject(this.theta);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void addTrajectory(int state, int action, double reward){
        states.add(state);
        actions.add(action);
        rewards.add(reward);
    }

    public void load(){
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream("./saved_agent/thetaREINFORCE.txt"))) {
            this.theta = (double[]) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

}



