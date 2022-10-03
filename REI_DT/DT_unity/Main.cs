using System.Collections;

using System.Collections.Generic;
using UnityEngine;

using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.IO;
using System.Text;
using UnityEngine.UI;
using System.Text.RegularExpressions;

public class Main : MonoBehaviour
{

    Dictionary<int, float> x_origin1 = new Dictionary<int, float>()
    {
        {1, 0.288f},
        {2, 0.313f},
        {3, 0.331f},
        {4, 0.296f},
        {5, 0.304f}
    };

    GameObject[] jointArray = new GameObject[7];
    
    int steps = 1;
    int cnt = 0;
    static int mark1 = 1;
    bool moveTag = false;
    static string datasave;

    float pospastx = 0.0f;
    float pospasty = 0.0f;
    float pospastz = 0.0f;
    public int[] axis_number = new int[] { 2, 1, 1, 1, 2, 1 };
    private float[] jointAngleRealTime = new float[7] { 0, 0, 0, 0, 0, 0, 0 };

    public static float[] targetAngle = new float[7];

    public static List<float[]> datalist = new List<float[]>() {new float[1] { 0 }};
    public static string dataTosave;
  
    private float[] deltaAngle = new float[7];
    private float[] currentAngle = new float[7];
    private Thread receiveThread; 
    private TcpClient localClient; 
    
    float T = 0.033f; 
    static int orginmark = 1;
    private float f_forece_est = 0;
    private float x_displacement = 0; 
    private float x_displacement_mea = 0; 
    private float velocity = 0; 
    
    private float beta = 100; 
    private float gamma = 2; 
    float forgetfactor = 0;
    private float[] theta = new float[] { 0, 0 };
    private float[] K_kalman = new float[] { 0, 0 };
    private float[,] P_cov = new float[2, 2] { { 2, 0, }, { 0, 2 } }; 
    
    Vector3 ra = new Vector3(-0.398f, -0.281f, 0.035f);
    Vector3 rb = new Vector3(-0.488f, -0.066f, 0.055f);
    Vector3 rc = new Vector3(-0.240f, -0.358f, 0.035f);
    
    private Texture2D lineTex;
    private Texture2D frontTex;
	
    void InitObject()
    {
        string objName;
        for (int i = 0; i < jointArray.Length; i++)
        {
            objName = string.Format("joint{0}", i); 
            jointArray[i] = GameObject.Find(objName);
        }
    }

    void OnApplicationQuit()
    {
        
        receiveThread.Abort();
        localClient.Close();
        
    }

    public List<float[]> datastrTolist(string datastring, List<float[]> datalistpre1, int mark)
    {
        string dataStrpro;
        string[] msg;
        List<float[]> datalist1 = new List<float[]>();
        dataStrpro = datastring.Substring(2,datastring.Length-4);
        dataStrpro = dataStrpro.Replace('[',' ');
        msg = Regex.Split(dataStrpro,"],");   
        
        foreach (string data in msg)
        {
            float[] data1 = Array.ConvertAll<string, float>(data.Split(','), s => float.Parse(s));
            datalist1.Add(data1);
        }
        if (mark > 0)
        {
            for (int i = 0;i < datalist1.Count;i++)
            {
                if (datalist1[i].Length==1 && datalist1[i][0]==0)
                {
                    if (i < 5)
                    {
                        datalist1[i] = datalistpre1[i];
                    }
                    else
                    {
                        datalist1.RemoveAt(5);
                        for (int j = 5;j < datalistpre1.Count;j++)
                        {
                            datalist1.Add(datalistpre1[j]);
                        }   
                    }
                }
            } 
        }

        return datalist1;
    }

    private void SocketReceiver()
    {
        string head_sizeStr;
        string resultStr;
        byte[] head_data = new byte[5];
        List<float[]> datalistpre = new List<float[]>(); 
        int mark = 0;
        if (localClient != null)
        {
            while (true)
            {
                if (localClient.Client.Connected == false)
                    break;
                localClient.Client.Receive(head_data);
                head_sizeStr = Encoding.UTF8.GetString(head_data);
                
                int head_size=Convert.ToInt32(head_sizeStr.Replace('"',' '));
                
                byte[] resultBuffer = new byte[head_size];
                localClient.Client.Receive(resultBuffer);
                resultStr = Encoding.UTF8.GetString(resultBuffer);
                
                
                datalist = datastrTolist(resultStr,datalistpre,mark);
               
                datalistpre = datalist;
                mark+=1;
                
                int d = 0;
                foreach (float angle in datalist[0])
                {
                        targetAngle[d] = angle*180/ (float)(Math.PI);
                        d++;
                }
                targetAngle[6] = datalist[3][0];   
                getDeltaAngle();  
                cnt = 0;
                moveTag = true;

            
            }
        }
    }
    
    private void InitClientSocket()
    {
        string sHostIpAddress = "192.168.1.115";
        
        int nPort = 30004;
        
        
        IPAddress ipAddress = IPAddress.Parse(sHostIpAddress);        
        
        localClient=new TcpClient();  
        try 
        {
            localClient.Connect(ipAddress, nPort);
            receiveThread = new Thread(SocketReceiver);
            receiveThread.Start();
            Debug.Log("客户端-->服务端完成,开启接收消息线程");
        }
        catch (Exception ex)
        {
            Debug.Log("客户端异常：" + ex.Message);
        } 
        Debug.Log("连接到服务器 本地地址端口:" + localClient.Client.LocalEndPoint + "  远程服务器端口:" + localClient.Client.RemoteEndPoint);     
    }
   
    public void setAngle(float[] angle)
    {
        jointArray[0].transform.GetComponent<GeneralJoint>().angleMove(-angle[0]);
        jointArray[1].transform.GetComponent<GeneralJoint>().angleMove(-angle[1] - 90);
        jointArray[2].transform.GetComponent<GeneralJoint>().angleMove(-angle[2]);
        jointArray[3].transform.GetComponent<GeneralJoint>().angleMove(-angle[3] - 90);
        jointArray[4].transform.GetComponent<GeneralJoint>().angleMove(-angle[4]);
        jointArray[5].transform.GetComponent<GeneralJoint>().angleMove(-angle[5]);
        jointArray[6].transform.GetComponent<Gripper>().positionMove(angle[6]);
        updateJointAngle();
    }

    void updateJointAngle()
    {
        for (int i = 0; i < 7; i++)
        {
            jointAngleRealTime[i] = currentAngle[i];
            
        }
    }
    void getDeltaAngle()
    {

        for (int i = 0; i < deltaAngle.Length; i++)
        {
            deltaAngle[i] = (targetAngle[i] - jointAngleRealTime[i]) / steps;
        }
    }

    void getCurrentAngle()
    {
        for (int i = 0; i < 7; i++)
        {
            currentAngle[i] = jointAngleRealTime[i] + deltaAngle[i];
            
        }
    }

    void SPRLS(float x_position, float velocity_t2, float f_force_mea, int orginmark1)
    {
      
        float x_origin = x_origin1[orginmark1];
        f_force_mea = -f_force_mea;
        
        velocity_t2 = -velocity_t2;
        
        
        
        x_displacement_mea = (x_origin - x_position);
        float[] phi = new float[] { velocity_t2, x_displacement_mea };
        
        
        float[] error1 = new float[2];
        float error2 = 1;
        for (int i = 0; i < 2; i++)
        {
            error1[i] = P_cov[i, 0] * phi[0] + P_cov[i, 1] * phi[1];
            error2 += phi[i] * error1[i];

        }
        for (int i = 0; i < 2; i++)
        {
            K_kalman[i] = error1[i] / error2;
        }

        
        float error = f_force_mea - (theta[0] * phi[0] + theta[1] * phi[1]); 
        for (int i = 0; i < 2; i++)
        {
            theta[i] += K_kalman[i] * error;
        }
       
        float error4 = f_force_mea - (theta[0] * phi[0] + theta[1] * phi[1]); 
        forgetfactor = gamma * (float)Math.Pow(error4, 2.0);
        float[] error3 = new float[2];
        float[,] error3_1 = new float[2, 2] { { 0, 0 }, { 0, 0 } };
        for (int i = 0; i < 2; i++)
        {
            error3[i] = phi[0] * P_cov[0, i] + phi[1] * P_cov[1, i];
            error3_1[0, i] = K_kalman[0] * error3[i];
            error3_1[1, i] = K_kalman[1] * error3[i];
            P_cov[0, i] -= error3_1[0, i];
            P_cov[1, i] -= error3_1[1, i];
            P_cov[i, i] += beta * (forgetfactor >= 100 ? forgetfactor : 0);
        }
        
        velocity = velocity_t2;
        x_displacement = x_displacement_mea;
        f_forece_est = theta[0] * velocity + theta[1] * x_displacement;
        wr.WriteLine(Convert.ToString(theta[0]) + " , " + Convert.ToString(theta[1]) + " , " + Convert.ToString(x_position) + " , " + Convert.ToString(velocity_t2) + " , " + Convert.ToString(f_force_mea) + " , " + Convert.ToString(f_forece_est) + " , " +Convert.ToString(x_origin) + " , " +  Convert.ToString(forgetfactor) + "\n");
        
        Debug.Log("阻尼：" + theta[0] + "," + "刚度：" + theta[1]);
    }
    
    void Start()
    {
        
        InitObject(); 
        InitClientSocket();

    }

    void FixedUpdate()
    {
        if (moveTag == true)
        {
            
            if (cnt < steps)
            {

                getCurrentAngle();  
                
                setAngle(currentAngle); 
                
                cnt++;
            }
            else
            {

                setAngle(targetAngle);
                
                moveTag = false;
            }
        }

    }
}
