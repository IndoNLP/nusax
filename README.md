# NusaX

NusaX is a high-quality  multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak.

NusaX is created by translating existing sentiment analysis dataset into local languages.
Our translations are written and verified by local native speakers. Therefore, NusaX can be broken down into 2 separate tasks:

- [NusaX-MT: Machine Translation task](https://github.com/IndoNLP/nusax/tree/main/datasets/mt)
- [NusaX-Senti: Sentiment Analysis task](https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment)

Additionally, we also release the [NusaX-Lexicon](https://github.com/IndoNLP/nusax/tree/main/datasets/lexicon), which consists of parallel, sentiment lexicon of 10 Indonesian local languages.

## Research Paper
You can find the details in [our paper](https://aclanthology.org/2023.eacl-main.57.pdf). The paper was awarded an **Outstanding Paper** at EACL 2023.

If you use our dataset or any code from this repository, please cite the following:
```
@inproceedings{winata-etal-2023-nusax,
    title = "NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages",
    author = "Winata, Genta Indra  and
      Aji, Alham Fikri  and
      Cahyawijaya, Samuel  and
      Mahendra, Rahmad  and
      Koto, Fajri  and
      Romadhony, Ade  and
      Kurniawan, Kemal  and
      Moeljadi, David  and
      Prasojo, Radityo Eko  and
      Fung, Pascale  and
      Baldwin, Timothy  and
      Lau, Jey Han  and
      Sennrich, Rico  and
      Ruder, Sebastian",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.57",
    pages = "815--834"
}
```

## License
The dataset is licensed with CC-BY-SA, and the code is licensed with Apache-2.0.
