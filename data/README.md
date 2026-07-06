# Data

## Amazon Reviews 2023 — Video Games

This project uses the Video Games category from the UCSD Amazon Reviews 2023 dataset.

| Attribute | Value |
|---|---|
| Interactions | ~500,000 user-product ratings |
| Users | 400,000+ |
| Items | 63,000 video games |
| Rating scale | 1–5 stars |
| Additional fields | Timestamps, verified purchase flags, helpful votes |

## Download

Source: [https://cseweb.ucsd.edu/~jmcauley/datasets.html](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

Under "Amazon Reviews 2023," select the **Video Games** category and download the ratings/reviews file.

## Setup

Raw data is not included in this repository due to file size. After downloading, place the file locally and update the `data_path` variable near the top of `recommender_system.ipynb` to point to its location.
