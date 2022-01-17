---
layout: post
title: 3D Scene Representation
author: Sidharth
published: true
---

## Implicit Representation ##
The word implicit means "suggested though not directly expressed". Following this nuance, in implicit representations the 3D information is not definitely expressed. Rather, in this method a neural network is assumed to learn the 3D information and its weights are the representation of the scene. So this forms the ideal recipe for supervised learning where we learn the 3D information by feeding the network with the scene information represented using the camera parameters. Although it seems to be a perfect way to represent a scene, it has its own de-merits.

Before exploring its efficiency, we will look at NeRF (Neural Radiance Fields) {% cite NERF %} as an example for implicit representation. In this work, a deep fully connected network learns the color $$( \begin{bmatrix} R &G &B \end{bmatrix}^T \in \mathbb{R}^3)$$ and density $$(\sigma \in \mathbb{R})$$ information of the scene as a function of the location of a point $$\begin{bmatrix} x &y &z \end{bmatrix}^T$$ along a ray and the viewing direction of the scene $$\begin{bmatrix} \theta &\phi \end{bmatrix}^T$$. In practice, viewing direction is expressed as a 3D unit vector. The network is learnt by overfitting on a scene i.e. showing images of the same scene captured from different views. During inference, the network regresses the color and density information when queried with a novel camera location and direction.

![_config.yml]({{ site.baseurl }}/images/Nerf_2.PNG)
*Figure 1 : A point on the ray along with its viewing direction as shown in (a) is passed through $$( F_{\theta} )$$ which is trained to regresses the color and density information as shown in (b). This regressed density and color which vary along the ray (c) is now passed through a volume rendering network to generate an image of the scene which satisfies the underlying viewing direction $$( \begin{bmatrix} \theta &\phi \end{bmatrix}^T )$$. The entire network including $$( F_{\theta} )$$ and rendering network is trained using the rendering loss which is the squared $$(L_2)$$ norm between the generated image and the the ground truth image (d).* 

To intuitively understand the output color and density information given by the network, consider a camera ray passing through the scene as shown in Figure 1. The final color information that we see on the image formed is all embedded in this ray. The final color contributed by this ray is the accumulation of all the color surfaces this ray passes through before reaching the camera. And on its way to the camera the ray can pass through different types of transparent, opaque and translucent surfaces. The network essentially tries to trace out the path through which the ray has travelled and this path can be expressed using the color and density information. The output color gives the RGB information of a point $$\begin{bmatrix} x &y &z \end{bmatrix}^T$$ on the path while the output density is the probability of a surface being present at that point. Hence, by combining the color and density information we can reconstruct the scene.

Although using this approach we get the complete information of the scene, the downside to this approach is its inference time. For querying one input we would need an forward pass through the deep MLP network. And hence explicit representations were intriduced to address this issue.

{% bibliography --cited %}
