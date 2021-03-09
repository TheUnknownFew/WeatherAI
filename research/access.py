"""
| **Author:** Alexander Cherry
| **Author Email:** Alexander.Pennstate@yahoo.com
| **Made:** 2/1/2021
| **Last Modified:** 2/3/2021 by Alexander Cherry
| **Purpose:** General Data overview, Current uses, and Future considerations.
|
| **Preface:**
| This documentation shall explain what data is being used, where the data comes from, and how the data is being used.
  This documentation aims to be as apparent as possible, and explain the thought process behind decisions made in regard
  to data. Personal research into the topics discussed is recommended.
|
| As this document provides a lot of different resources, it is recommended that any individual viewing them does their
  own exploration of the web sites provided.
|
----------------------------------

|
| **Data Source Overview:**
|
| **Top Domain:** https://www.noaa.gov/
| Future exploration of this project suggests to use data from the National Oceanic and Atmospheric Administration and
  all of its associated organizations. the NOAA is a wealth of data. Currently, this project only uses a tiny fraction
  of data available by NOAA and all of its associate and sub organizations. The following are the primary and secondary
  sources explored by this project thus far:
- Primary sources are sources of data that have been heavily considered thus far and have data that are being used by
  the system. NOT all data is used by the primary source. Primary sources MAY have a multitude of other available data
  that could be considered for future use.
- Secondary sources are sources of data that have been considered but are not currently in use by the system. Future
  considerations of if data should be pulled from such sources is required. Secondary source may also act to provide
  supplemental data for data that is already in use.

| **Primary source(s) of data:**
|
|    **NOAA ESRL** - Earth System Research Laboratories
|    NOAA Earth System Research Laboratories is an organization of many laboratories. ESRL consists of four primary
     laboratories. Under ESRL are the Chemical Sciences (CSL), Global Monitoring (GML), Global
     Systems (GSL), and Physical Sciences (PSL) Laboratories.
|
|    **Domain:**                 https://www.esrl.noaa.gov
|    **About:**                  https://www.esrl.noaa.gov/about/
|    **Research Site Explorer:** https://www.esrl.noaa.gov/gmd/dv/site/
|    **Dataset Explorer:**       https://www.esrl.noaa.gov/gmd/dv/data

|    For the sake of this project, only data from the Global Monitoring Laboratory has been considered thus far. Future
     work on this system may want to consider data from the other laboratories. Current implementation only focuses on
     atmospheric data in respect to climate change. The other laboratories may be able to provide different data in
     respect to climate change. Additionally, the Global Monitoring Laboratory provides many different research sites
     that provide data aside from atmospheric data. For more information, please check the research site explorer. A list
     of sites and what kind of data they provide will be listed there. A more comprehensive list of every single dataset
     provided by the Global Monitoring Laboratory is listed in the dataset explorer.
|
|    The current implementation of this system fetches data from: - browse the dataset explorer for more datasets
|    (Any other locations where data is being fetched from should be listed here)
|
|    **key:**
|    *Fetching* - The current system currently fetches data from...
|    *Supplemental* - The data is not in use, but the system may want to consider fetching from in the future...
|
|    **ftp access:**             ftp://aftp.cmdl.noaa.gov/
|        **Fetching:**           Atmospheric Data (CH4, CO2, N2O, SF6)
|        **From:**               ftp://aftp.cmdl.noaa.gov/data/greenhouse_gases/
|        **Fetching:**           Meteorological Data
|        **From:**               ftp://aftp.cmdl.noaa.gov/data/meteorology/in-situ/
|        **Supplemental:**       trace gases
|        **From:**               ftp://aftp.cmdl.noaa.gov/data/trace_gases/
|
| **Secondary source(s) of data:**
|
|    **NOAA NCDC** - National Climatic Data Center *(under consideration)*
|    **Domain:**                 https://www.ncdc.noaa.gov
|    **Climate at a Glance:**    https://www.ncdc.noaa.gov/cag/
|
----------------------------------

|
| Data Outlook:
|
----------------------------------

|
| System Outlook:
|
----------------------------------

|
| **Useful / Exploratory Resources and future extension research:**
| The following could be used for a wider scope to the project, and/or to do personal research to learn topics covered by
  this project.
|
- **NOAA Climate** - https://www.climate.gov/about
- **US Environmental Protection Agency** - https://www.epa.gov/aboutepa
- **Intergovernmental Panel on Climate Change** - https://www.ipcc.ch/about/ \n
  **Data Distribution Center:** https://www.ipcc-data.org/
- **Catalogue of Data** - https://catalog.data.gov/dataset
|  *WARNING:* This site is not limited to data of interest by this project. Additional searching through this
   site is required.
"""
from ftplib import FTP
from urllib import request

from bs4 import BeautifulSoup
from pandas import DataFrame

SITE_META_CSV = 'data/esrl_site_meta.csv'
"""
Relative path from access.py to the save location of the ESRL research site information table.
"""
CACHED_DATASETS = 'data/cached_datasets.json'
"""
Relative path from access.py to the save location of cached_datasets.json.
"""


def update_research_sites():
    """
    | **Author:** Alexander Cherry
    | **Author Email:** Alexander.Pennstate@yahoo.com
    | **Made:** 2/1/2021
    | **Last Modified:** 2/3/2021 by Alexander Cherry
    |
    | **Overview:**
    | -----------------------------
    | Site codes and names are used by the ESRL FTP in order to distinguish file names. Caching this information allows
      for the ease of access to construct FTP file paths to specific data.
      A * in the table means a site or project has been discontinued. This information may still be useful, however.
    |
    | **Function Documentation:**
    | -----------------------------
    | Fetches site codes with corresponding site names from 'https://www.esrl.noaa.gov/gmd/dv/site/?program=all'.
      Site codes and names are cached to to 'data/esrl_site_meta.csv'. This file contains site codes for all sites and all
      ESRL projects; not necessarily every site that pertains to the system's current implementation.
    |
    | Call this function to update site table. Calls to this function should be infrequent, and made only during periodic
      checks to the NOAA website for updated information.
    """
    with request.urlopen('https://www.esrl.noaa.gov/gmd/dv/site/?program=all') as site_explorer:
        soup = BeautifulSoup(site_explorer.read(), features='html.parser')
    site_table = soup.find(id='table')
    columns = [col_name.text.strip() for col_name in site_table.thead.find_all('th')]
    table_content = []
    for row in site_table.tbody.find_all('tr'):
        col_vals = [val.text.strip() for val in row.find_all('td')]
        table_content.append(col_vals)
    site_table = DataFrame(table_content, columns=columns)
    site_table.to_csv(SITE_META_CSV)


def fetch_dataset_from_ftp(ftp_path: str, file_name: str):
    """

    :return:
    """
    pass


def update_cached_datasets():
    """

    :return:
    """
    # Todo: Implement
    pass


def load_cached_datasets():
    """

    :return:
    """
    # Todo: Implement
    pass


if __name__ == '__main__':
    pass
