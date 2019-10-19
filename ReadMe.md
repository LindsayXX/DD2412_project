Replication track in [NeurIPS 2019 Reproducibility Challenge](https://reproducibility-challenge.github.io/neurips2019/)
of [Learning where to look: Semantic-Guided Multi-Attention Localization for Zero-Shot Learning](https://arxiv.org/abs/1903.00502)

This project hasn't finished yet, more information will be update soon!

### 1. Multi-Attention Subnet: 
1. VGG19(?) backbone  <br /> 
 input: Image  <br /> 
 output: Features representation  <br /> 
2. K-means  <br /> 
    input: Feature representation <br /> 
    output: 2 groups of feature representation <br /> 
3. Global averge pooling + 2 Fully connected layers(ReLU) + Sigmoid <br /> 
input: 2 groups of feature representation  <br /> 
(intermediate result: Channel descriptor p1, p2)  <br /> 
output: Channel-wise attention weight vector a1, a2 <br /> 
4. Weighted-sum <br /> 
input: Feature representation, channel-wise attention weight <br /> 
output: 2 Attention maps <br /> 

#### 2. Region Cropping Subnet:  
1. f_CNet(2 fully connected layers) <br /> 
input: Attention maps <br /> 
output: [t_x, t_y, t_s] <br /> 
2. Boxcar Mask (cropping operation x o V_i)  <br /> 
input: Attention maps, [t_x, t_y, t_s] <br /> 
output: masked images (x_i^part) <br /> 

#### 3. Joint Feature Learning Subnet
1. VGG backbone + Global average pooling <br /> 
input: original image/ masked image <br /> 
output: visual feature vector $\theta$ <br /> 
2. Transformation: <br /> 
input: visual feature vector <br /> 
output: Semantic feature vector <br /> 

#### 4. Classification
different between seen and unseen classes <br /> 

### Reference:
[Pedro Morgado and
Nuno Vasconcelos. Semantically consistent regularization
for zero-shot recognition. In CVPR, 2017.](https://arxiv.org/abs/1704.03039)

[Feng Wang, Xiang Xiang, Jian Cheng,
and Alan Loddon Yuille. Normface: l 2 hypersphere embedding for face verification. In ACMMM. ACM, 2017.](https://arxiv.org/abs/1704.06369)

[Jianlong Fu, Heliang Zheng, and Tao Mei.
Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition.
In CVPR, pages 4438–4446, 2017.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)

[Heliang Zheng, Jianlong Fu, Tao Mei,
and Jiebo Luo. Learning multi-attention convolutional
neural network for fine-grained image recognition. In
ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_Multi-Attention_Convolutional_ICCV_2017_paper.pdf)
