### 1. Multi-Attention Subnet: 
1. VGG19(?) backbone  <br /> 
 input: Image  <br /> 
 output: Features representation  <br /> 
2. K-means  <br /> 
    input: Feature representation <br /> 
    output: 2 groups of feature representation <br /> 
3. Global averge pooling + 2 Fully connected layers(ReLu) + Sigmoid <br /> 
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

####4. Classification <br /> 
different between seen and unseen classes <br /> 

