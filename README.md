# Deep Learning Based Pedestrian Trajectory Prediction In Urban Environment Using Time-To-Collision and Crossing Zones Indicators
Welcome to the GitHub page of my final project in MSc Intelligent Systems, Afeka. 
Under the supervision of **Prof. Erel Avineri** and advisory of **Dr. Yehudit Aperstein**. <br />
<img width="357" height="86" alt="image" src="https://github.com/user-attachments/assets/0487ef42-bba7-4c80-b580-9168b78dde27" />


## Repository Guide
The project is based on the inD dataset: [https://levelxdata.com/ind-dataset/](https://levelxdata.com/ind-dataset/)

The TTC calculation script is taken from: [https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision](https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision)

The project contains the following folders:
**data_preprocessing** for data preprocssing.
**models/v2** for creating the deep learning models, for training, loading, evaluating and analysis of results.

## The Project's Abstract
Vulnerable road users and vehicle conflicts are a major cause of fatalities in urban areas, underscoring the need to improve advanced driver assistance systems (ADAS) and autonomous vehicles behavior in this field. This work addresses the challenge of predicting pedestrian motion in naturalistic urban environment, focusing on edge conflict situations, which is a critical task in safety systems. 
A deep learning-based pedestrian trajectory prediction (PTP) framework is proposed to incorporate novel data-derived features including Time-To-Collision (TTC) as a surrogate safety measure and crossing zone indicators as contextual cues. Previous literature has extensively explored deep learning approaches for pedestrian trajectory prediction with diverse model types and scenes, however most trajectory prediction works in the urban domain have not specifically focused on conflicts between pedestrians and vehicles or incorporated TTC as an interaction feature or crossing zones as contextual cues. As far known to this date, this kind of work was yet to be tested.
The PTP is formulated as a sequence-to-sequence prediction task. Several neural network models are proposed. The models are trained and evaluated using a dataset characterized by natural, heterogeneous, and large-scale urban traffic scenarios. In the proposed methodology, each deep learning model is evaluated twice: once using only positional features (xy coordinates) and once with the addition of TTC and crossing zone features. This dual evaluation allows us to assess the impact of incorporating TTC and crossing zones on the performance of the prediction. The models are assessed using standard metrics: ADE and FDE as well as a new evaluation approach that counts the same metrics exceeding a safety threshold.
Results demonstrate that inputting TTC and crossing zone features to a LSTM model with pooling mechanism, improves prediction accuracy and reduces the frequency of large safety-critical errors relative to a basic LSTM model with position only as input. The findings suggest that utilizing pooling layers for both interaction and environmental context is essential for better PTP in real-world urban settings.

## Quick brief of the project
### Data
<img width="624" height="313" alt="image" src="https://github.com/user-attachments/assets/d0235c59-a87a-466b-b0a0-3ff44b1e11fb" />
<img width="498" height="269" alt="image" src="https://github.com/user-attachments/assets/e718f7dc-5c19-467c-84f1-c15167bb64bd" />


### Methodology
The suggested innovation is to add a meaningful interaction feature and scene context feature to the data. The added features will focus and emphasize learning on the interesting interactions between pedestrians and vehicles on interesting areas in the junction. It is assumed that in this way the conflicts will be quantified and marked so the model will perform better in predicting trajectories, in particular evasive motions. 

<img width="400" height="105" alt="image" src="https://github.com/user-attachments/assets/5bf5bbd8-f311-4a59-911c-d8d0410e48a4" />


### Interaction and Environment Modeling
<img width="538" height="543" alt="image" src="https://github.com/user-attachments/assets/59b1a10e-bcb2-4435-976e-3fa7138a39c2" />

### LSTM With Pooling architecture
<img width="404" height="230" alt="image" src="https://github.com/user-attachments/assets/4e3729da-f385-4ece-a9d8-1299b51705ab" />


### Experiments
Comparison of LSTM with pooling with xy+ttc+zones input vs. LSTM with only xy input


<img width="712" height="640" alt="image" src="https://github.com/user-attachments/assets/62b4a752-b00a-49d2-891f-8568f1b4441c" />
