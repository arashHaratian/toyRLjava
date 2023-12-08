package org.RL.envs;

import java.util.Map;
import java.util.Random;

public class CorridorEnv {
    private Random random = new Random();
    private int currentRealState;
    private Map<Integer, Double> realStateRewards;
    private Map<Integer, Integer> realStateObsState;
    private boolean done;
    private double probDirectDone;
    private int obsShape, actionShape;
    public CorridorEnv(){
        obsShape = 4;
        actionShape = 2;
        this.currentRealState = 2;
        this.realStateObsState = Map.of(
                1,0,
                2, 1,
                3,2,
                4, 0,
                5, 2,
                6, 3,
                7, 0);
        this.realStateRewards = Map.of(
                1,-1.0,
                2, -0.1,
                3,-0.1,
                4, 1.0,
                5, -0.1,
                6, -0.1,
                7, -1.0);
        this.done = false;
        this.probDirectDone = 0.01;
    }
    public void reset(){
        this.currentRealState = 2;
    }

    public void step(int action){
        if (random.nextFloat() < this.probDirectDone){
            this.currentRealState = 7;
            return;
        }
        this.currentRealState = action == 0 ? this.currentRealState - 1 : this.currentRealState + 1;
    }
    public boolean isDone() {
        int obsState = this.realStateObsState.get(this.currentRealState);
        return obsState == 0;
    }

    public int getState() {
        return this.realStateObsState.get(this.currentRealState);
    }

    public double getReward() {
        return this.realStateRewards.get(this.currentRealState);
    }

    public double getProbDirectDone() {
        return probDirectDone;
    }

    public void setProbDirectDone(double probDirectDone) {
        this.probDirectDone = probDirectDone;
    }

    public int getActionShape() {
        return actionShape;
    }

    public int getObsShape() {
        return obsShape;
    }
}
