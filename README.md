# HyperVID
基于深度学习移动端车型识别，支持1776种常见车辆品牌及子品牌。停车场卡口跟二手车图片TOP1准确率85%左右，TOP5 99%以上，自然场景车辆准确率略低，我们训练数据主要基于停车场卡口图片。使用者可以基于我们的框架进行修改，后续我们会进一步优化模型，提高准确率与速度，并丰富类型库。

Vehicle Type Identification which supporting for 1776 kinds of models. Part of Data from Second-hand car trading service website.

[Models Table](label.txt)

#### 特性

- 速度快，基于mobilenet设计，更适合移动端部署

- 种类丰富，能够支持1776种常见车型

- 准确度高，TOP1 准确率超过85%

- 非常适合停车场卡口，结合车牌识别一同部署


#### 识别测试APP

- 体验 Android APP：[https://fir.im/HyperVID](https://fir.im/HyperVID)


#### Demo Image on Android

![demo](demo.png)

#### TODO

- 丰富车型库

- 进一步优化模型，提高速度，降低模型大小


#### Related Dataset

[BIT-Vehicle Dataset](http://iitlab.bit.edu.cn/mcislab/vehicledb/)

[MIT Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

[The CompCars dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
