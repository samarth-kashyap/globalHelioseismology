# Global Helioseismology

This python package enables the computation of helioseismic cross-spectra for time-series.
- Currently supports only data from Helioseismic Magnetic Imager (HMI).

## Installation
First clone the repository using `git clone https://github.com/samarth-kashyap/globalHelioseismology`. Enter the cloned repository: `cd globalHelioseismology` and install the package using `pip install -e .`. After this, you should be able to `import globalHelioseismology` in your python script.

### Citing
The development of this code was initiated through the work [Kashyap and Bharati Das et. al. (2021)](https://arxiv.org/abs/2101.08933). If you use this code, please cite using

```
@ARTICLE{Kashyap-2021-ApJS,
       author = {{Kashyap}, Samarth G. and {Das}, Srijan Bharati and {Hanasoge}, Shravan M. and {Woodard}, Martin F. and {Tromp}, Jeroen},
        title = "{Inferring Solar Differential Rotation through Normal-mode Coupling Using Bayesian Statistics}",
      journal = {\apjs},
     keywords = {Helioseismology, Solar oscillations, Solar differential rotation, Markov chain Monte Carlo, 709, 1515, 1996, 1889, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {253},
       number = {2},
          eid = {47},
        pages = {47},
          doi = {10.3847/1538-4365/abdf5e},
archivePrefix = {arXiv},
       eprint = {2101.08933},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..253...47K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
