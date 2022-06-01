# NusaX

NusaX is a high-quality  multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak.

NusaX is created by translating exsiting sentiment analysis dataset into local languages.
Our translations are written and verified by local native speakers. Therefore, NusaX can be broken down into 2 separate tasks:

- [NusaX-MT: Machine translation task](https://github.com/IndoNLP/nusax/tree/main/datasets/mt)
- [NusaX-Senti: Sentiment Analysis task](https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment)

Additionally, we also release a [NusaX-Lexicon](https://github.com/IndoNLP/nusax/tree/main/datasets/lexicon), which consists of parallel, sentiment lexicon of 10 Indonesian local languages.

## Research Paper
You can find the details in [our paper](https://arxiv.org/pdf/2205.15960.pdf)

If you use our dataset or any code from this repository, please cite the following:
```
@misc{winata2022nusax,
      title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages}, 
      author={Genta Indra Winata and Alham Fikri Aji and Samuel Cahyawijaya and Rahmad Mahendra and Fajri Koto and Ade Romadhony and Kemal Kurniawan and David Moeljadi and Radityo Eko Prasojo and Pascale Fung and Timothy Baldwin and Jey Han Lau and Rico Sennrich and Sebastian Ruder},
      year={2022},
      eprint={2205.15960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
The dataset is licensed with CC-BY-SA, and the code is licensed with Apache-2.0.
