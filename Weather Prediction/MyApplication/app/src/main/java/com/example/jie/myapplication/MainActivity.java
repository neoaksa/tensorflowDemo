package com.example.jie.myapplication;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Arrays;
public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button submitButton = (Button) findViewById(R.id.submit_button);
        submitButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                JSONObject postData = new JSONObject();
                JSONArray jsonArray1 = new JSONArray();
                JSONArray jsonArray2 = new JSONArray();
                JSONArray jsonArray3 = new JSONArray();
                float[][] dummy = new float[12][10];
                for(float d[]:dummy) {
                    Arrays.fill(d, (float) 0.0);
                    for(float item:d) {
                        try {
                            jsonArray1.put(item);
                            jsonArray2.put(item);
                            jsonArray3.put(item);
                        }catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                }

                try {

                    postData.put("day1", jsonArray1);
                    postData.put("day2", jsonArray2);
                    postData.put("day3", jsonArray3);
                    new SendDeviceDetails().execute("http://104.211.28.38:5000/predict", postData.toString());
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        });
    }
    private class SendDeviceDetails extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            String data = "";
            HttpURLConnection conn = null;
            try {
                URL url = new URL(params[0]);
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json;charset=UTF-8");
                conn.setRequestProperty("Accept","application/json");
                conn.setDoOutput(true);
                conn.setDoInput(true);
                Log.i("JSON", params[1]);
                DataOutputStream os = new DataOutputStream(conn.getOutputStream());
                os.writeBytes(params[1]);
                os.flush();
                os.close();
                Log.i("STATUS", String.valueOf(conn.getResponseCode()));
                Log.i("MSG" , conn.getResponseMessage());
                InputStreamReader inputStreamReader = new InputStreamReader(conn.getInputStream());
                int inputStreamData = inputStreamReader.read();
                while (inputStreamData != -1) {
                    data += (char) inputStreamData;
                    inputStreamData = inputStreamReader.read();
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                conn.disconnect();
            }
            return data;
        }
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            Log.e("TAG", result); // this is expecting a response code to be sent from your server upon receiving the POST data
        }
    }
}