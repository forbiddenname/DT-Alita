using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Gripper : MonoBehaviour {

    //private string objectName;
    //private int joint_number;
    //GameObject camera1;
    //Main mainScript;
    GameObject[] finger= new GameObject[2];
    float f_p;
    //float[] temp_angle = { 0, 0, 0 };
    //int axis_number;//该关节的旋转轴，0代表x轴，1代表y轴，2代表z轴，均为本体轴

    //public int Axis_number
    //{
    //    get
    //    {
    //        return axis_number;
    //    }
    //    set
    //    {
    //        axis_number = Axis_number;
    //    }

    //}

  
//读取该关节的当前关节转角
//public float read_tempangle()
//{
//    return temp_angle[Axis_number];
//}
//改变该关节的当前关节转角
//public void changetemp_angle(float angle)
//{
//    temp_angle[Axis_number] = -angle;
//    transform.localRotation = Quaternion.Euler(temp_angle[0], temp_angle[1], temp_angle[2]);//欧拉转动

//}

//通过控制函数控制该关节的转角
public void positionMove(float pos)
    {
        //if (ang > 180)
        //{
        //    while (ang > 180)
        //    {
        //        ang -= 360;
        //    }
        //}
        //else if (ang < -180)
        //{
        //    while (ang < -180)
        //    {

        //        ang += 360;
        //    }

        //}
        //temp_angle[Axis_number] = pos/2;
        //f_p = (pos / 180 * (float)(Math.PI) / 10/86) /2  ;    //二指夹 真实:模型=86：100；机械臂 真实：模型=1:1
        f_p = (pos  / 10 / 86) / 2; //单位为m
        finger[0].transform.localPosition = new Vector3(-0.1044f,-f_p, 0);
        finger[1].transform.localPosition = new Vector3(-0.1044f, f_p, 0);
        //transform.localRotation = Quaternion.Euler(temp_angle[0], temp_angle[1], temp_angle[2]);
    }

    //void angle_change(ref float temp, float step)
    //{
    //    temp = temp + step;
    //}


    // Use this for initialization
    void Start()
    {


      
        finger[0]= gameObject.transform.Find("finger0").gameObject;
        finger[1]= gameObject.transform.Find("finger1").gameObject;
        //camera1 = GameObject.Find("Main Camera");
        //objectName = gameObject.name;
        //joint_number = int.Parse(objectName.Substring(5, 1));//提取joint1的1这个字符，转为int类型

        //mainScript = camera1.GetComponent<Main>();
        //axis_number = mainScript.axis_number[joint_number];//该gameobject代表哪个方向的转动，返回的是0,1,2,

    }

    // Update is called once per frame
    void Update()
    {

    }
}
