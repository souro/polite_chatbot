# Polite Chatbot: A Text Style Transfer Application

This repo contains the code and data of the paper: [Polite Chatbot: A Text Style Transfer Application](https://aclanthology.org/2023.eacl-srw.9/).

## Overview

Our method: We (1) train the politeness transfer model; (2) generate synthetic training data by applying the transfer model to neutral utterances; (3) train the dialogue models using the synthetic data.


<p align="left">
  <img src="image/Polite_Chatbot_Arch.png"/>
</p>

## Walkthrough

### Dependency

    pip install -r requirements.txt

*Will add more information in this section soon.*

## Citing
If you use this data or code please cite the following:
  
    @inproceedings{mukherjee-etal-2023-polite,
    title = "Polite Chatbot: A Text Style Transfer Application",
    author = "Mukherjee, Sourabrata  and
      Hude{\v{c}}ek, Vojt{\v{e}}ch  and
      Du{\v{s}}ek, Ond{\v{r}}ej",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-srw.9",
    pages = "87--93",
}

## License

    Author: Sourabrata Mukherjee
    Copyright © 2023 Sourabrata Mukherjee.
    Licensed under the MIT License.

## Acknowledgements

This research was supported by Charles University projects GAUK 392221, GAUK 302120, and SVV 260575, and by the European Research Council (Grant agreement No. 101039303 NG-NLG).
