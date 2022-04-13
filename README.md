# Automatic scoring of COVID-19 severity in X-ray
In this project, we propose a two-stage workflow used for the segmentation and scoring of lung diseases. The workflow inherits quantification, qualification, and visual assessment of lung diseases on X-ray images estimated by radiologists and clinicians. It requires the fulfillment of two core stages devoted to lung and disease segmentation as well as an additional post-processing stage devoted to scoring. The latter integrated block is utilized, mainly, for the estimation of segment scores and computes the overall severity score of a patient. The models of the proposed workflow were trained and tested on four publicly available X-ray datasets of COVID-19 patients and two X-ray datasets of patients with no pulmonary pathology.  Based on a combined dataset consisting of 580 COVID-19 patients and 784 patients with no disorders, our best-performing algorithm is based on a combination of DeepLabV3+, for lung segmentation, and MA-Net, for disease segmentation. The proposed algorithms’ mean absolute error (MAE) of 0.30 is significantly reduced in comparison to established COVID-19 algorithms; BS-net and COVID-Net-S, possessing MAEs of 2.52 and 1.83 respectively. Moreover, the proposed two-stage workflow was not only more accurate but also computationally efficient, it was approximately 11 times faster than the mentioned methods. In summary, we proposed an accurate, time-efficient, and versatile approach for segmentation and scoring of lung diseases illustrated for COVID-19 and with broader future applications for pneumonia, tuberculosis, pneumothorax, amongst others.

## Data
### Stage I: Lung Segmentation
Table 1. Description of the datasets used for lung segmentation

|                                   **Dataset**                                   | **Training** | **Validation** | **Testing** |  **Total**  |
|:-------------------------------------------------------------------------------:|:------------:|:--------------:|:-----------:|:-----------:|
|    [Darwin](https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset)     |     4884     |      611       |     611     | 6106 / 90%  |
| [Montgomery](https://www.kaggle.com/raddar/tuberculosis-chest-xrays-montgomery) |     110      |       14       |     14      |  138 / 2%   |
|   [Shenzhen](https://www.kaggle.com/raddar/tuberculosis-chest-xrays-shenzhen)   |     452      |       57       |     57      |  566 / 8%   |
|                                      Total                                      |  5446 / 80%  |   682 / 10%    |  682 / 10%  | 6810 / 100% |

### Stage II: Disease Segmentation and Scoring
Table 2. Description of the datasets used for COVID-19 segmentation and scoring

|                                  **Dataset** 	                                  | **COVID-19** 	 | **Normal** 	 | **Training** 	 | **Validation** 	 | **Testing** 	 | **Total**  	  |
|:-------------------------------------------------------------------------------:|:--------------:|:------------:|:--------------:|:----------------:|:-------------:|:-------------:|
|    [ACCD](https://github.com/agchung/Actualmed-COVID-chestxray-dataset)    	    |   49      	    |   0     	    |   39      	    |    5       	     |   5      	    |  49 / 4%   	  |
| [CRD](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)     	 |   104     	    |   0     	    |   83      	    |    10       	    |   11     	    |  104 / 8%  	  |
|        [CCXD](https://github.com/ieee8023/covid-chestxray-dataset)    	         |   399     	    |   0     	    |   319     	    |    40       	    |   40     	    | 399 / 29%  	  |
|     [FCXD](https://github.com/agchung/Figure1-COVID-chestxray-dataset)    	     |   28      	    |   0     	    |   22      	    |    3       	     |   3      	    |  28 / 2%   	  |
|    [CXN](https://www.kaggle.com/paultimothymooney/chest-xray-pneumoni)     	    |    0      	    |   431    	   |   344     	    |    43       	    |   44     	    | 431 / 31%  	  |
|    [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)    	     |    0      	    |   353    	   |   282     	    |    35       	    |   36     	    | 353 / 26%  	  |
|                                   Total    	                                    |  580 / 43%  	  | 784 / 57% 	  | 1089 / 80%  	  |  136 / 10%   	   | 139 / 10%  	  | 1364 / 100% 	 |

## Methods
### Stage I: Lung Segmentation
![Stage I](media/stage_1.png "Stage I")

### Stage II: Disease Segmentation and Scoring
![Stage II](media/stage_2.png "Stage II")

### Post-processing: Score Estimation
![Post-processing](media/post_processing.png "Post-processing")