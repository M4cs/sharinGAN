# sharinGAN
A Generative Adversarial Network implementation that generates Sharingans. This was trained on Google Colab with 3000 epochs and it took about 10 minutes to train.

# Requirements
- PyTorch
- numpy

# Dataset
I scraped bunch of images from different websites and then resized them to 128x128 size. This dataset consists of 56 images and this was the reason why it took 10 minutes to train.

# Here's the Generated output
![final image](https://github.com/jaychandra6/sharinGAN/blob/master/static/final.png)

# Why did I write this code?
I started learning PyTorch from the past 10 days. Fortunately, I found a very intuitve playlist on YouTube by deeplizar. Honestly, those videos are dope. [Here's the link.](https://www.youtube.com/playlist?list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG) I wanted to test out with something interesting.

I found GANs when I was taking up Andrew Ng's Introduction to Machine Learning Course and wanted to implement it in PyTorch but, I don't know how to implement it in PyTorch. So, I searched for GAN implementation on YouTube and found a video by [Ayush Chaurasia](https://www.youtube.com/c/AyushChaurasia) where he implemented [Generative Adversarial Network](https://www.youtube.com/watch?v=aZpsxMZbG14) from the original reasearch paper in PyTorch. His video is the only reason why this code is here and it's very similar to his code.

I then trained the GAN with FashionMNIST dataset and got pretty good results in 30 epochs but I wanted to test it out on a different dataset. Then I searched on Kaggle but most of the datasets are above 300 Mb. So, I decided not to download them because I am too lazy. Then as usual I was bingewatching YouTube and found a video by jabrils where he implements Auto Encoders. He uploaded pokemon dataset to kaggle which is about 80 Mb and I downloaded it and trained it on Google Colab. After 300 iterations the result looks like trash. The new generated pokemon can only be seen when you imagine its shape and edges xD. I left this project for 2 days until I found a video about GAN's in my suggestions on YouTube. The channel name is [AngryCoder](https://www.youtube.com/channel/UCta6mmYG1NLeDeFFaLP2eug). He trained a GAN and it generates Sharingans. I then went to the description box and did not find any link to his code. So, I thought that I should implement this. I scraped images of Sharingans from couple of websites then resized them to 128x128 and uploaded as a zip to Google Colab. In fact, I took the name of this repo from his video.

Honestly, if I did not find these YouTube videos, I would've never implemented this code. Huge shoutout to them.
Here are the channel links:
[deeplizard](https://www.youtube.com/channel/UC4UJ26WkceqONNF5S26OiVw)
[AngryCoder](https://www.youtube.com/channel/UCta6mmYG1NLeDeFFaLP2eug)
[Ayush Chaurasia](https://www.youtube.com/c/AyushChaurasia)
