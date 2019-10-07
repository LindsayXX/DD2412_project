###1. Multi-Attention Subnet: 
1. VGG19(?) backbone
 input: Image 
 output: Features representation 
2. K-means 
    input: Feature representation
    output: 2 groups of feature representation
3. Global averge pooling + 2 Fully connected layers(ReLu) + Sigmoid
input: 2 groups of feature representation
(intermediate result: Channel descriptor p1, p2)
output: Channel-wise attention weight vector a1, a2
4. Weighted-sum
input: Feature representation, channel-wise attention weight
output: 2 Attention maps

####2. Region Cropping Subnet: 
1. f_CNet(2 fully connected layers)
input: Attention maps
output: [t_x, t_y, t_s]
2. Boxcar Mask (cropping operation x o V_i) 
input: Attention maps, [t_x, t_y, t_s]
output: masked images (x_i^part)

####3. Joint Feature Learning Subnet
1. VGG backbone + Global average pooling
input: original image/ masked image
output: visual feature vector $\theta$
2. Transformation:
input: visual feature vector
output: Semantic feature vector

####4. Classification
different between seen and unseen classes

