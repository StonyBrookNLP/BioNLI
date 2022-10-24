## What is BioNLI?
### BioNLI is a biomedical NLI dataset using controllable text generation


This is the official page for the paper <a href='#'> BioNLI: Generating a Biomedical NLI Dataset Using Lexico-semantic Constraints for Adversarial Examples </a>,  accepted at EMNLP2022 (Findings).
 <!-- You can find our paper <a href='https://arxiv.org/abs/2205.04652'> here </a> -->

<!-- Mohaddeseh Bastan, Nishant Shankar, Mihai Surdeanu, Niranjan Balasubramanian.  -->

BioNLI is the first dataset in biomedical natural language inference. This dataset contains abstracts from biomedical literature and mechanistic premises generated with nine different strategies. 

### Example
In the following example we see an example of an entry in the BioNLI dataset. Some supporting text was removed to save space. The premise is a set of sentences talking about two biomedical entiteis. The consistent hypothesis is the original conclusion sentence from the abstract paper, the inconsistent hypothesis is the generated sentence with one of the different nine strategies.

### Coming Soon ###


<!-- <img src="assets/img/dataexample_v3.drawio.svg" alt="Image of SuMe stats"/> -->

### Dataset Statistics

There are two different versions of this dataset. One is the large distribution which contains all possible perturbations and the other is the balanced distirbution. They both share the same test set. For the full distribution, we generate as many perturbations as possible for dev and test set, but for training each instance is perturbed once.
#### Full Distribution:

<img src="assets/img/full.png" alt="Image of full stats" width="50%" height="#"/>


#### Balanced Distribution:

<img src="assets/img/balanced.png" alt="Image of balanced stats" width="40%" height="#" />




### Download the data

The dataset can be downloaded here:

The full set can be downloaded from <a href="https://drive.google.com/drive/folders/1-wNvAYs4ULJFNkeVHaUJM3U7slX-vtiB?usp=sharing">here</a>.

The balanced set can be downloaded from <a href="https://drive.google.com/drive/folders/187W4RCk1cJnKKO95NPljxZYelUgMOHf8?usp=sharing">here</a>. 

To access the test set please contact <a href = "mailto: mbastan@cs.stonybrook.edu">me</a>.

### License

BioNLI is distributed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>. 


### Liked us? Cite us!

Please use the following bibtex entry:
```

```

