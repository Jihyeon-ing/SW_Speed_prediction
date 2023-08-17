# Three-day Forecasting of Solar Wind Speed Using SDO/AIA Extreme-ultraviolet Images by a Deep-learning Model

Here, a deep learning model for solar wind speed for the next 3 days at near the Earth is introduced.

DOI: 10.3847/1538-4365/ace59a

CITATION: Jihyeon Son et al 2023 ApJS 267 45

## DATA 
- Input data:
  - SDO/AIA 211 and 193 images (previous 5 days with 12h cadnece, total 10 images, resized to 64x64)
  - OMNI web solar wind speed (previous 5 days with 6h cadence, total 20 porints) 
- Target data: OMNI web solar wind speed for the next 72h with 6h cadence (total 12 points)

## MODEL ARCHITECTURE
