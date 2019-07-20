# mn-tweet-bert
## Introduction
Using this repo, you can classify whether the input text is similar to 
[@Iderodcomedian](https://twitter.com/Iderodcomedian)'s or [@enkhbat](https://twitter.com/enkhbat)'s. 
And you will fine-tuning will take at most an hour.


## How to use
### Data
For fine-tuning, the small noisy data was crawled from twitter (tweets of @Iderodcomedian and @enkhbat).

You can download it using

`source scripts/get_train_data.sh`

### Models
You can also use the fine-tuned model on the small dataset using:<br>
`wget https://storage.googleapis.com/bucket-97tsogoo-gmail/mn-tweets-idree-enkhbat/models/tweetmn-epoch-10/pytorch_model.bin && mv pytorch_model.bin ./output/`

If you want to fine-tune the pre-trained BERT (Mongolian) from scratch you can use:

`./scripts/run_classifier.sh`

and this will output the evualation scores too. Before that you need [pre-trained BERT on Mongolian text corpus](https://github.com/tugstugi/mongolian-bert).

### Results
After the training, you will see similar to:

```
eval_loss = 0.5363224036991596
global_step = 290
loss = 7.182709101972908e-05
tweetmn = {'acc': 0.9, 'f1': 0.888888888888889, 'acc_and_f1': 0.8944444444444445}
```

### Visualization

Pytorch tool in [jessevig/bertviz](https://github.com/jessevig/bertviz) can be used to see 
attention scores after fine-tuned model is trained. 

Comment: Originally in the bertviz, sentencepiece implementation is not released yet!


![Attention-head example](https://github.com/bayartsogt-ya/mn-tweet-bert/raw/master/images/Screen%20Shot%202019-07-20%20at%209.03.55%20PM.png)

From the picture above, we can see could learn some grammatical features and interesting connections
between words.

For example the words `компанийнхаа` and `менежерүүд` were similar to phrase `компаний менежер`
and attention scored high on words not working as the phrase in the current sentence.


## References

Based on following repositories:
- [google-research/bert](https://github.com/google-research/bert/)
- [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-transformers)
- [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese)
- [tugstugi/mongolian-bert](https://github.com/tugstugi/mongolian-bert)
- [jessevig/bertviz](https://github.com/jessevig/bertviz)
