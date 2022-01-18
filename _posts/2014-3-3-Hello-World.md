---
layout: post
title: 3D Scene Representation
author: Sidharth Nair, Meher Abhijeet
published: true
---

Rendering multiview images of a scene has been a long standing research area in the field of cumputer vision. Given multiple images of a scene with varying viewing directions, the task of multiview image synthesis is to generate photorealistic images of the scene with novel views.

![_config.yml]({{ site.baseurl }}/images/Nerf_1.PNG)
*Figure 1 : Given multiple input images of a scene with varying viewing directions, a multi view rendering network ( in this case {% cite NERF %} ) renders new views of the scene.*

Before rendering an image, we first try to represent the scene using an intermediate representation that provides some 3D information. Based on the characteristics of these representations they are divided into Explicit, Implicit and Hybrid. In this blog we try to intuitively explain Explicit, Implicit and Hybrid representations as a build up to explaining the method proposed in Efficient Geometry-aware {3D} Generative Adversarial Networks.

## Implicit Representation ##
The word implicit means "suggested though not directly expressed". Following this nuance, in implicit representations the 3D information is not definitely expressed. Rather, in this method a neural network is assumed to learn the 3D information and its weights are the representation of the scene. So this forms the ideal recipe for supervised learning where we learn the 3D information by feeding the network with the scene information represented using the camera parameters. Although it seems to be a perfect way to represent a scene, it has its own de-merits.

Before exploring its efficiency, we will look at NeRF (Neural Radiance Fields) as an example for implicit representation. In this work, a deep fully connected network learns the color $$( \begin{bmatrix} R &G &B \end{bmatrix}^T \in \mathbb{R}^3)$$ and density $$(\sigma \in \mathbb{R})$$ information of the scene as a function of the location of a point $$\begin{bmatrix} x &y &z \end{bmatrix}^T$$ along a ray and the viewing direction of the scene $$\begin{bmatrix} \theta &\phi \end{bmatrix}^T$$. In practice, viewing direction is expressed as a 3D unit vector. The network is learnt by overfitting on a scene i.e. showing images of the same scene captured from different views. During inference, the network regresses the color and density information when queried with a novel camera location and direction.

![_config.yml]({{ site.baseurl }}/images/Nerf_2.PNG)
*Figure 1 : A point on the ray along with its viewing direction as shown in (a) is passed through $$( F_{\theta} )$$ which is trained to regresses the color and density information as shown in (b). This regressed density and color which vary along the ray (c) is now passed through a volume rendering network to generate an image of the scene which satisfies the underlying viewing direction, $$\begin{bmatrix} \theta &\phi \end{bmatrix}^T $$. The entire network comprising of $$F_{\theta}$$ and rendering network is trained using the rendering loss which is the squared $$L_2$$ norm between the generated image and the the ground truth image (d).* 

To intuitively understand the output color and density information given by the network, consider a camera ray passing through the scene as shown in Figure 1. The final color information that we see on the image formed is all embedded in this ray. The final color contributed by this ray is the accumulation of all the color surfaces this ray passes through before reaching the camera. And on its way to the camera the ray can pass through different types of transparent, opaque and translucent surfaces. The network essentially tries to trace out the path through which the ray has travelled and this path can be expressed using the color and density information. The output color gives the RGB information of a point $$\begin{bmatrix} x &y &z \end{bmatrix}^T$$ on the path while the output density is the probability of a surface being present at that point. Hence, by combining the color and density information we can reconstruct the scene.

Although using this method we get the complete information of the scene, the downside to this approach is its inference time. For querying one input we would need an forward pass through the deep MLP network. And hence explicit representations were introduced to address this issue.

## Explicit Representation ##

Unlike the implicit representation explained above, in this method we express 3D data using explicitly defined shapes ( spheres, cubes, cuboids etc.. ). In this approach, we represent the surface by learning a function $$( f )$$ which maps input images to a 3D grid $$ f : \mathbb{R}^2 \rightarrow \mathbb{R}^3 $$. An example of such a 3D representation is Voxel grid. Voxels are essentially 3D pixels shaped in the form of perfect cubes.

![_config.yml]({{ site.baseurl }}/images/voxel_1.jpeg)

In theory, voxel grid is a fast technique for modelling complex surfaces. We can accurately replicate real world objects by employing a proper rendering technique armed with a very high resolution voxel grid. To understand how a voxel grid represents a 3D scene, consider the chair shown in figure. If we were to approximate this chair's surface using the 3D voxel grid, in the most basic way we fill up each voxel of the grid with a $$ 1 $$ or $$ 0 $$ to represent whether or not a part of the chair surface is present at this voxel location. Hence using this binary method to populate the voxel grid, we can form a very coarse way to approximate the chair's surface. To also represent the finer surface textures, we can increase the number of voxels in the voxel grid there by increasing the number of voxels corresponding to a particular point on the chair's surface. 

![_config.yml]({{ site.baseurl }}/images/voxel_2.jpg)

This approach addresses the inference time problem associated with the implicit representation because in this case we can query the voxel grid in $$ O(1) $$ time. However this method is inefficient in terms of memory usage as voxel grid requires $$O(V^3)$$ memory for a side of length $$V$$.
