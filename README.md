# Three-day Forecasting of Solar Wind Speed Using SDO/AIA Extreme-ultraviolet Images by a Deep-learning Model

Here, a deep learning model for solar wind speed for the next 3 days at near the Earth is introduced.

+ DOI: 10.3847/1538-4365/ace59a </br>
+ Paper:  [Son+ (2023)](https://iopscience.iop.org/article/10.3847/1538-4365/ace59a). </br>
+ CITATION: Jihyeon Son et al 2023 ApJS 267 45
</br>

## DATA 
- Input data:
  - SDO/AIA 211 and 193 images (previous 5 days with 12h cadnece, total 10 images, resized to 64x64)
  - OMNI web solar wind speed (previous 5 days with 6h cadence, total 20 points) 
- Target data: OMNI web solar wind speed for the next 72h with 6h cadence (total 12 points)
</br>

## MODEL ARCHITECTURE
![architecture_2](https://github.com/Jihyeon-ing/SW_Speed_prediction/assets/96173406/c3b5b670-c381-48da-9c47-e2b0ff0da2e3)
