# tensorflow_wgangp_brain

TODO:
Adding checkpoints/models download link for inference. 
Adding brain texture generation jupyter notebook. 


Installation:
pip install -r requirement.txt
# we are using tensorflow 1.10.0 version for deformation GAN

Inference:
please refer to the infererence.sh

Training:
please refer to run_gResnetDDCGAN.sh

BRAIN_TEXTURE_GENERATION:
coded in tensorflow 2.2.0 version  
https://www.tensorflow.org/tutorials/generative/pix2pix.  

We employ padding and uniform scaling to 256x256 resolution in input and resampled back to original resolution.  
flipping, crop and resize strategies is not employed.  


