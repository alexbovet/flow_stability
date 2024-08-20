# Flow Stability in ants

This is an exemplary application of `flowstab`.

## Data

We use the [publicly available data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.65q64) from the paper [Short-term activity cycles impede informatoin transmission in ant colonies](https://doi.org/10.1371/journal.pcbi.1005527).

## Log

- Getting the data with:
  ```bash
  wget https://datadryad.org/stash/downloads/file_stream/8341 -O Contacts.tar.gz
  ```
- Extracting the content to the folder `contacts`:
  ```bash
  mkdir -p data/contacts
  tar -xf Contacts.tar.gz -C data/content
  ```
  This extracts 16 \*.txt files which are actually csv files with space as delimiter.

- Run the analysis with `python analyze_ant_iteractions.py`
