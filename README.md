# TGA-DGAFPN
Time-Leaping Gradient Aligned Dynamics Gramian Angle Feature Pyramid Network (TGA-DGAFPN)
This git is used to support model parameters and is mainly used to display the main framework of the model
用于支撑模型参数的Git, 主要用于展示模型的主框架。

Table 1: Parameters of the Layers of the Proposed Model
Layer/Module	Type/Components	Parameters	Output Shape
Input	1D Signal	channels=1, length=2048	(B, 1, 2048)
DTLGADF	Deformable Conv1d	in=1, out=4, k=7, offset_scale=0.4	(B, 4, 2048)
bn1	BatchNorm1d	num_features=4	(B, 4, 2048)
MaxPool1d	Pooling	k=4, s=4	(B, 4, 512)
layer1	Bottleneck1D ×2	in=4, out=8, stride=1	(B, 8, 512)
layer2	Bottleneck1D ×1	in=8, out=16, stride=2	(B, 16, 256)
layer3	Bottleneck1D ×1	in=16, out=32, stride=2	(B, 32, 128)
layer4	Bottleneck1D ×1	in=32, out=64, stride=2	(B, 64, 64)
Top-layer	Conv1d	in=64, out=64, k=1, s=1, p=0	(B, 64, 64)
Latlayer1	Conv1d	in=64, out=64, k=1, s=1, p=0	(B, 64, 64)
Latlayer2	Conv1d	in=32, out=64, k=1, s=1, p=0	(B, 64, 128)
Latlayer3	Conv1d	in=16, out=64, k=1, s=1, p=0	(B, 64, 256)
Upsample P5→P4	Interpolate	mode="linear"	(B, 64, 128)
P4 Fusion	Element-wise Add	P5↑ + Lateral1(C4)	(B, 64, 128)
Upsample P4→P3	Interpolate	mode="linear"	(B, 64, 256)
P3 Fusion	Element-wise Add	P4↑ + Lateral2(C3)	(B, 64, 256)
Upsample P3→P2	Interpolate	mode="linear"	(B, 64, 512)
P2 Fusion	Element-wise Add	P3↑ + Lateral3(C2)	(B, 64, 512)
smooth3	Conv1d	in=64, out=32, k=3, s=1, p=1	(B, 32, 512)
dropout	Dropout	p=0.3	(B, 32, 512)
AdaptiveAvgPool1d	Pooling	output_size=1	(B, 32, 1)
classifier	Linear	in=32, out=num_classes	(B, num_classes)
