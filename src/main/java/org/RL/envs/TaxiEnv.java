package org.RL.envs;

import org.bytedeco.cpython.*;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;


public class TaxiEnv {

    PyObject globals;
    private boolean done;
    private double state, reward;
    private int actionShape, obsShape;
    private String renderText;

    private TaxiEnv() throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        Py_Initialize(org.bytedeco.gym.presets.gym.cachePackages());
        _import_array();
        this.globals = PyModule_GetDict(PyImport_AddModule("__main__"));
    }

    public TaxiEnv(String envName) throws Exception {
        this();
        String str = """
                import gym
                env = gym.make("%s", render_mode = "ansi")
                state, _ = env.reset()
                action_shape = env.action_space.n
                obs_shape = env.observation_space.n
                done = False
                reward = 0
                """.formatted(envName);


        PyRun_StringFlags(str, Py_file_input, this.globals, this.globals, null);
        this.checkPyErr();
        this.updateParams();
        setObsShape();
        setActionShape();
    }



    private void checkPyErr(){
        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }

    private void updateParams(){
        this.updateState();
        this.updateDone();
        this.updateReward();
        this.checkPyErr();
    }

    private void updateState() {
        this.state = PyFloat_AsDouble(PyDict_GetItemString(this.globals, "state"));
    }

    private void updateDone() {
        this.done = PyFloat_AsDouble(PyDict_GetItemString(this.globals, "done")) != 0;
    }
    private void updateReward() {
        this.reward = PyFloat_AsDouble(PyDict_GetItemString(this.globals, "reward"));
    }

    private void updateRender() {
        String str = """
                render_text = env.render()
                """;

        PyRun_StringFlags(str, Py_file_input, this.globals, this.globals, null);


        this.renderText = PyBytes_AsString(PyUnicode_AsASCIIString(PyDict_GetItemString(this.globals, "render_text"))).getString();
//        this.renderText = PyBytes_AsString(PyObject_Str(PyDict_GetItemString(this.globals, "render_text"))).getString();
        this.checkPyErr();
    }

    private void setActionShape() {
        this.actionShape = (int) PyFloat_AsDouble(PyDict_GetItemString(this.globals, "action_shape"));
    }

    private void setObsShape() {
        this.obsShape = (int) PyFloat_AsDouble(PyDict_GetItemString(this.globals, "obs_shape"));
    }
    public void reset(){
        String str = """
                state, _ = env.reset()
                done = False
                reward = 0
                """;

        PyRun_StringFlags(str, Py_file_input, this.globals, this.globals, null);
        this.updateParams();
        this.checkPyErr();
    }

    public void step(int action){
        String str = """
                state, reward, terminated, truncated, info = env.step(%d)
                done = terminated or truncated
                """.formatted(action);

        PyRun_StringFlags(str, Py_file_input, this.globals, this.globals, null);
        this.updateParams();
        this.checkPyErr();
    }

    public int getState(){
        return (int) this.state;
    }

    public boolean isDone(){
        return this.done;
    }
    public double getReward(){
        return this.reward;
    }

    public void render(){
        this.updateRender();
        System.out.println(this.renderText);
    }

    @Override
    public String toString() {
        return "State: " + getState() + "\t" +
                "Reward: " + getReward() + "\t" +
                "Done: " + isDone() + "\t";
    }

    public int getActionShape() {
        return actionShape;
    }

    public int getObsShape() {
        return obsShape;
    }


}
