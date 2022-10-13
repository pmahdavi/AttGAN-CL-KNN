# AttGAN-CL-KNN

This is our implementation for CSE 586 final project (conditional text to image generation). Our code is built on the top of the [AttnGAN+CL](https://github.com/huiyegit/T2I_CL). 

# Loading the data, and pretrained weights
Data should be downloaded from [metadata](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ), and saved to `data/`, and also download imaged from [birds image data](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images) and put it in `data/birds/`

For pretrained image and text encoder, you should download the files from [pretrained](https://drive.google.com/file/d/15w_mKV7UzmC3jMqplKyMawUEEJaJozTZ/view?usp=sharing), and save it to `DAMSMencoders/`

# Traning our algorithm
For traning you can run `python main.py --cfg cfg/bird_attn4.yml --gpu 0 --optim adam_np` if you want to train with adam_nm optimizer. Otherwise, you can run `python main.py --cfg cfg/bird_attn4.yml --gpu 0` 

# Evaluation

For generating the fake images from the validation dataset, run `python main.py --cfg cfg/eval_bird.yml --gpu 0`. You can also save your own sentence in "./data/birds/example_captions.txt", and set the B_VALIDATION to False in eval_bird.yaml and then run `python main.py --cfg cfg/eval_bird.yml --gpu 0` to get the generated images. 

To evaluate the FID score, you can run the following command: 
`python fid_score.py --gpu 0 --batch-size 50 --path1 real_images_bird --path2 fake_images_bird`
