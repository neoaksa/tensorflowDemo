package com.example.jie.tensoronandorid;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/weather.pb";
    private static final String INPUT_NODE = "input";
    private static final String[] OUTPUT_NODES = {"y_"};
    private static final String OUTPUT_NODE = "y_";
    private static final long[] INPUT_SIZE = {1,10};
    private static final int OUTPUT_SIZE = 2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        /** One time initialization: */
        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(getApplicationContext().getAssets(),
                                                    MODEL_FILE);

        /** Continuous inference (floats used in example, can be any primitive): */
        // this should be same as model
        float input[] = new float[10];
        Arrays.fill(input,0);

        float[] output = new float[OUTPUT_SIZE];
        // loading new input
        // INPUT_SHAPE is an int[] of expected shape, input is a float[] with the input data
        tensorflow.feed(INPUT_NODE,input ,INPUT_SIZE);

        // running inference for given input and reading output
        tensorflow.run(OUTPUT_NODES);
        // output is a preallocated float[] in the size of the expected output vector
        tensorflow.fetch(OUTPUT_NODE, output);
    }
}
