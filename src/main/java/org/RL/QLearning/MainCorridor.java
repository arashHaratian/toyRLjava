package org.RL.QLearning;

import org.RL.QLearning.QLearningAgent;
import org.RL.envs.CorridorEnv;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class MainCorridor {
    public static void main(String[] args) {

        ArrayList<Double> trainingRes = trainAgent();
        XYChart trainPlot = plot(trainingRes);
        new SwingWrapper<XYChart>(trainPlot).displayChart();
    }

    public static ArrayList<Double> trainAgent(){
        final int ITERATIONS = 1_000;
        CorridorEnv env;
        boolean done;
        int currentState, nextState, action;
        double reward, returns = 0;

        ArrayList<Double> returnsList = new ArrayList<>();

        try {
            env = new CorridorEnv();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        QLearningAgent agent = new QLearningAgent(env.getObsShape(), env.getActionShape(), 0.5, 0.99, 0.01);


        for (int i = 0; i < ITERATIONS; i++) {
            env.reset();
            currentState = env.getState();
            done = false;
            reward = 0;
            returns = 0;

            while(!done){
                action = agent.chooseEpsGreedyAction(currentState);
                env.step(action);
                nextState = env.getState();
                reward = env.getReward();
                done = env.isDone();

                agent.update(currentState, action, nextState, reward);

                returns += reward;
                currentState = nextState;

            }
            returnsList.add(returns);

            if(i % 1 == 0)
                System.out.println(i + "  :  return = "  + returns + "  ,  avgReturn"  + returnsList.stream().mapToDouble(val -> val).average());
        }

        agent.save();

        return(returnsList);
    }
    public static XYChart plot(ArrayList<Double> trainingRes){
        var xData = IntStream.range(0,trainingRes.size()).boxed().toList();
        XYChart chart = new XYChartBuilder().width(1600).height(1200).title("The Learning Process\n").xAxisTitle("Episode Number").yAxisTitle("Sum of Reward").build();
        chart.getStyler().setChartTitleVisible(true);
        chart.addSeries("Reward of The Episode", xData, trainingRes);
        return chart;
    }
}
