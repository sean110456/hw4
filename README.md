# hw4

## File content
* vdsr.py :The pytorch vdsr net, built by https://github.com/twtygqyy/pytorch-vdsr
* hw4.py: Code for training the network
* hw4-output.py: Code for output the 3x scale high resolution image, using the model trained by hw4.py

## Run
python hw4.py
python hw4-output.py

## How the high resolution images are generated
1. Upscale the image by 3 and convert it to YCbCr format
2. Throw the Y channel of the YCbCr image to the trained model
3. Combined the new Y channel with the original Cb Cr channels
