# Hotel Booking Demand Dataset

## Provenance & Attribution
- **Canonical Citation:** Antonio, N., de Almeida, A., and Nunes, L. (2019). *Hotel booking demand datasets*. **Data in Brief**, 22, 41–49. [https://doi.org/10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126)
- **Public URL:** `https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv`
- **Coverage:** July 1, 2015 – August 31, 2017 (26 months).
- **Scope:** 119,390 observations across 32 initial variables representing two Portuguese hotel properties:
  - **H1 (Resort Hotel):** 40,060 records from the Algarve coastal resort region.
  - **H2 (City Hotel):** 79,330 records from metropolitan Lisbon.

## Automated Ingestion
If `hotels.csv` is ever deleted or missing locally, running `python main.py`, `python src/data_loading.py`, or launching `streamlit run app.py` automatically detects its absence and downloads it directly from the public GitHub repository.
