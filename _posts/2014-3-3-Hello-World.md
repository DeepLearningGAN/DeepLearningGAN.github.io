---
layout: post
title: You're up and running!
published: true
---
---
layout : post
title : 3D Scene Representation
---

Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).

![_config.yml]({{ site.baseurl }}/images/config.png)

## Implicit Representation ##
The word implicit means "suggested though not directly expressed". Following this nuance, in implicit representations the 3D information is not definitely expressed. Rather in this method the neural network is assumed to learn the 3D information and its weights are the representation of the scene. So this forms the ideal recipe for supervised learning where we learn the 3D information by feeding the network with the scene information represented using the camera parameters. Although it seems to be a perfect way to represent a scene, it has its own merits and de-merits.

Let us understand how this works before exploring its efficiency. We will use NeRF (Neural Radiance Fields) as an example for implicit representation. In this work, they use a deep fully connected network to learn the color $(\in \mathbb{R}^3)$ and density $(\in \mathbb{R})$ information of the scene as a function of the location $\begin{bmatrix} x &y &z \end{bmatrix}^T$ on a ray and its orientation $\begin{bmatrix} \theta &\phi \end{bmatrix}^T$.  **Provide Examples**. To intuitively understand the output color and density information given by the network, consider a camera ray passing through the scene as shown in figure \ref{fig:nerf2}. The final color information that we see on the image formed is all embedded in this ray. The final color contributed by this ray is the accumulation of all the color surfaces this ray passes through before reaching the camera. And on its way to the camera the ray can pass through different types of transparent, opaque and translucent surfaces. The network essentially tries to trace out the path through which the ray has travelled and this path can be expressed using the color and density information. The output color gives the RGB information of every point along the path while the output density is the probability of a surface being present at every point along the path. Hence, by combining the color and density information we can essentially reconstruct the scene.
