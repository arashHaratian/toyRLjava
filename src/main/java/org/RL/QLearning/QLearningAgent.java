package org.RL.QLearning;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.Random;

public class QLearningAgent {
    private double[][] QTable;
    final private double LR, GAMMA, EPS;
    private Random random = new Random();
    private int numObs, numActions;
    public QLearningAgent(int numObs, int numActions, double lr, double gamma, double epsilon){
        this.LR = lr;
        this.GAMMA = gamma;
        this.EPS = epsilon;
        this.numObs = numObs;
        this.numActions = numActions;
        this.QTable = new double[numObs][numActions];
    }

    public QLearningAgent(int numObs, int numActions){
        this(numObs, numActions, 0.1, 0.6, 0.1);
    }

    public void update(int currentState, int action, int nextState, double reward){
        double oldValue = this.QTable[currentState][action];
        double maxNextState = Arrays.stream(this.QTable[nextState]).max().getAsDouble();
        double newValue = (1 - this.LR) * oldValue + this.LR * (reward + this.GAMMA * maxNextState);
        this.QTable[currentState][action] = newValue;
    }

    public double[][] getQTable() {
        return this.QTable;
    }

    public int chooseBestAction(int currentState){
        int bestAction;
        double[] row = this.QTable[currentState];
        double maxVal = Arrays.stream(row).max().getAsDouble();
        List<Integer> maxIdx = IntStream.range(0, row.length).filter(i -> row[i] == maxVal).boxed().toList();

        if (maxIdx.size() > 1){
            bestAction = maxIdx.get(random.nextInt(maxIdx.size()));
        } else {
            bestAction = maxIdx.get(0);
        }

        return bestAction;
    }

    public int chooseEpsGreedyAction(int currentState){
        int action;
        if (random.nextFloat() < this.EPS){
            action = random.nextInt(this.numActions);
            return action;
        }
        action = chooseBestAction(currentState);
        return action;
    }

    public void save(){
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("./saved_agent/QTable.txt"))) {
            out.writeObject(this.QTable);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(){
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream("./saved_agent/QTable.txt"))) {
            this.QTable = (double[][]) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

}
