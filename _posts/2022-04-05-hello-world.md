---
title: 你好，世界！
date: 2022-03-26 10:34:00 +0800
categories: [随笔]
tags: [生活]
pin: true
author: 湾区书记汤姆

toc: true
comments: true
typora-root-url: ../../tomstillcoding.github.io
math: false
mermaid: true

image:
  src: /assets/blog_res/2021-03-30-hello-world.assets/huoshan.jpg
  alt: 签约成功

---

# 测距和大小估算～ 


首先在测距方面因为超声波测距仪只能得到2m之内的数据，所以我们选用了VL53L1x光学测距模块用来获得物体与测量仪之间的距离，而得到距离之后我们还要知道物体具体的大小，这时候，估算物体大小的算法有很多，我们用的是单目测距算法的引申，单目测距是由一个距离下色块像素的大小与距离乘积为一个常数k，用这个k通过除以图像像素数来估算物体与测量仪之间距离，而我们需要用的不是这个距离，我们已经有光学测距得到的距离且更加精确了。我们就拿一个模板的像素和距离积作为一个模值，当然该模值需要用该模块经过不同距离下得到积计算其均值，视觉处理测距测量物体体积,通过VL53L1X测距，乘以在图像中所呈现的像素点大小，得到一个基本保持恒定的常量 预估15000(可有dis*Lm得到）
，通过测量这个乘积的变化，模板设定的大小100mm*距离像素实际积的值/预估设定的值（15000）即可得到100*一个百分比，即表示现测量物体与100的大小比例关系，即可通过该方法得到其实际大小，用这个的方法即可得到其高度和半径，当然面积等即可得到。
```python
//代码片段
  h=int(100*b[3]*distance1/15250)
  w=int(100*b[2]*distance1/15250)
  r=int((h+w)/4) #半径取一半

```

# 颜色识别～ 
在颜色识别方面，OpenMv用的是LAB颜色阈值，不同于人眼正常的RGB和OpenCV中的HSV，L表示亮度，而A表示红绿阈值，B表示蓝色阈值和红绿中的黄色阈值，通过调整其得到想要的颜色。
thresholds=[ (0, 100, 24, 75, 14, 127),#red
            (0, 80, -128, -15, 0, 127)  ,#green
            (0, 100, -128, 49, -128, -10)] #blue
White_threshold=(66, 0, -128, 127, -128, 127)
这是大致的颜色阈值，三种不同颜色，而White_Threshold表示的是红蓝绿三种颜色交杂，用来计算其色块的像素，用上面方法得到其具体长度。而对于形状识别上面，我们选用的是OpenMV内置的霍夫圆检测img.find_circle，四元检测img.find_rects，线段检测等，通过霍夫圆检测即可得到我们想要的圆，矩形检测用来检测矩形，而我们又通过线段检测得到两条线段之间的角度差，如该值是50-70度之间表示是三角形，该方法可以较好的识别正三角形。
```python
//代码片段
for threshold_index in range(3):                         #通过索引遍历阈值
        blob1 = img.find_blobs([thresholds[threshold_index]], pixels_threshold=600)
        if blob1:
            for b in blob1:
                c=b.pixels()/(b.w()*b.h())#通过这个c即可用来分别是否为三角还是矩形，因为返回的像素是实际面积，而色块面积是包围最大矩形，而三角形是矩形的0.5倍数，即可用来分别矩形和三角形
                dis_angle=0
                n=0
                theta1 = 0
                theta2 = 0
                for lines in img.find_lines(roi=(b[0],b[1],b[2],b[3]), threshold = 1000, theta_margin = 25, rho_margin = 25):
                    theta1 = lines.theta()             #计算线段1的角度
                    n=n+1
                    if n==1:
                        theta2 = lines.theta()            #计算线段2的角度
                dis_angle=abs(theta1-theta2)
                if dis_angle>100:dis_angle=dis_angle-60   #考虑的是三角形倒过来的情况
                if(70>dis_angle>50  and 0.6>c>0.4 ):     #表示三角形
                    img.draw_rectangle(b.rect(), color = (0, 0, 0))
                    shape=3                              
                    color=threshold_index+1  #从索引中得到对应色块颜色 0红色 1 绿色 2 蓝色 加一后输出
                    if color ==1:
                     img.draw_string(60,10,'red',color=(255,0,0))
                    elif color ==2:
                      img.draw_string(60,10,'green',color=(0,255,0))
                    elif color ==3:
                       img.draw_string(60,10,'green',color=(0,0,255))
                    #通过索引得到其对应颜色,而对于圆形和矩形的颜色形状检测类似，由于有内置函数，所以不细写了，对于他们颜色可以通过判断LAB的上下限得到其对应颜色。

```

# 球体识别～ 
对于球体识别，可以考虑模块识别，比如ncc算法，FAST算法KCF算法，但是我们用tensorflow lite完成识别，而用google训练的经典模型，跑起来帧率非常低，非常影响运行速率。所以我们选择用edge impulse和teacheable machine 这两个软件去跑出训练模型tflite和labbels用来识别且达到不错的效果。由于需要按键区分，这个value0是GPIO0当其 感受到低电平后进入神经网络识别
```python
//代码片段
##########################引入tflite和label两个文件##############
net = "trained.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]
########################判断########################
   if value0 ==0:
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
       predictions_list = list(zip(labels, obj.output())) #通过labels和output两个列表组成字典
       maxp=predictions_list[0][1]
       i=0
       j=0
       for i in range(3):
            if predictions_list[i][1]>maxp:
             j=i
             maxp=predictions_list[i][1]
       if predictions_list[j][0] =='basketball':
         print("basketball")
         uart.write("basketball\r")
         a=1
       elif predictions_list[j][0] =='score ball':
         print("score ball")
         uart.write("score ball\r")
         a=2
       elif predictions_list[j][0] =='volleyball':
         print("volleyball")
         uart.write("volleyball\r")
         a=3
```

# 随动控制～ 
而在舵机随动控制中，我们选择用OpenMV连接舵机利用PID实现随动控制的效果，基本方式是寻找出色块中央，通过返回的色块中央与图像中心的偏差经过KP运算，再利用误差在时间上累积乘以KI得到控制量的大小，来控制舵机角度的变化。该步骤也是用按键控制，该按键是GPIO6用于识别功能，
```python
//代码片段
   if value6 ==0:
       blobs = img.find_blobs([Red_threshold])
       if blobs:
           max_blob = find_max(blobs)
           pan_error = max_blob.cx()-img.width()/2
           tilt_error = max_blob.cy()-img.height()/2
           print("tilt_error: ",tilt_error)

           img.draw_rectangle(max_blob.rect()) # rect
           img.draw_cross(max_blob.cx(), max_blob.cy()) # cx, cy

           pan_output=pan_pid.get_pid(pan_error,1)/2  #应该指定的是PID.PY中的self
           tilt_output=tilt_pid.get_pid(tilt_error,1)
           print("tilt_output: ",tilt_output)
           print("pan_output",pan_output)
           pan_servo.angle(pan_servo.angle()-pan_output)
           print("角度1",pan_servo.angle())
           uart.write("angle1:%d"%pan_servo.angle())
           tilt_servo.angle(tilt_servo.angle()-tilt_output)
           print("角度2",tilt_servo.angle())
           uart.write("angle2:%d \r\n"%tilt_servo.angle())
```
