[//]: # ( Should you not be filtering out the carrier frequency and not just filtering around it?)
[//]: # ( Filtering only makes sense if there are other signal components that you want to filter out or multiple signal components.)

# Modulation and Hilbert Transform Demo

## Usage
* Change the parameter values using the sliders at the top of the page.
* Show and hide the different components by clicking on the corresponding legend entries. 
* You can collapse different plots by clicking on the blue button above the plot.

## Modulation description

Define the carrier signal with frequency $f_c$ and amplitude $A$ as

$$
c(t) = A \sin(2 \pi f_c t)\,
$$

The modulating signal $m(t)$, with frequency $f_m$ (such that $f_m$ << $f_c$) is defined as 

$$
m(t) = M \cos\left(2\pi f_m t + \phi\right)= Ar \cos\left(2\pi f_m t + \phi\right),
$$

where $$r \in (0,1)$$ is the amplitude sensitivity (In an alternative formulation, $$M$$ is the amplitude of modulation).
Amplitude modulation occurs when the carrier $$c(t)$$    is multiplied by a positive signal $$(1 + m(t)/A)>0$$ :

$$
\begin{align}
  y(t) &= \left[1 + \frac{m(t)}{A}\right] c(t) \\
       &= \left[1 + r \cos\left(2\pi f_m t + \phi\right)\right] A \sin\left(2\pi f_c t\right)
\end{align}
$$

Using the trigonometric identities the resulting signal can be written as

$$
y(t) = A \sin(2\pi f_c t) + \frac{1}{2}Ar\left[\sin\left(2\pi \left[f_c + f_m\right] t + \phi\right) +  \sin\left(2\pi \left[f_c - f_m\right] t - \phi\right)\right].\,
$$

The modulated signal therefore ends up with three components in the frequency domain with sidebands having a reduced amplitude compared to the original modulating signal.

## Discussion
* If more elaborate filters are possible, it could make sense to first remove the carrier frequency and only allow the side bands in the filter sideband.
* The hilbert transform is only effective in extracting the envelope for very narrowband signals (i.e. a single frequency). This means that if other frequencies unrelated to the modulating signal are present, the envelope will be distorted.
* Filtering around the carrier frequency is mostly relevant if other signal components are present that are not related to the modulating signal that needs to be recovered. The reason for filtering around the carrier frequency is that this is where the sidebands are located that contain information about the modulating signal. 