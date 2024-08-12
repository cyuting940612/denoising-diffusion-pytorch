import pandas as pd
import os
import re
from tqdm import tqdm
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests, zipfile, io
import pyarrow as pa
import pyarrow.parquet as pq

path = os.getcwd()

# ------------------------------------------------------------------------------
bldg_count = 0

pumas = pd.read_csv(os.path.join(path, 'Data', 'houston pumas.csv')).astype(str)
pumas = 'G4800' + pumas
pumas = pumas.values.reshape(-1, ).tolist()

df = pd.read_csv(os.path.join(path, 'Data', 'TX_baseline_metadata_and_annual_results.csv'))
building = df['bldg_id'].loc[df['in.nhgis_puma_gisjoin'].isin(pumas)].astype(str).values.tolist()

options = Options()
options.add_argument('--headless')

driver = webdriver.Firefox(options=options)

# Iterating through dates
url = f'https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2023%2Fcomstock_amy2018_release_1%2Fmetadata_and_annual_results%2Fby_state%2Fstate%3DTX%2Fparquet%2F'
driver.get(url)

target_path = '//*[@id="tb-s3objects"]/tbody'

# Finding the table by its attribute which is 'aria-labelledby'
table = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, target_path)))

subfolders = driver.find_elements(By.XPATH, target_path + '/tr')

num_subfolders = len(subfolders)

# Iterate through the subfolders
for i in tqdm(range(1, num_subfolders + 1, 1)):
    # Construct the XPath for the link inside each subfolder
    link_xpath = target_path + f'/tr[{i}]/td[1]/a'
    link = driver.find_element(By.XPATH, link_xpath)

    file_url = link.get_attribute('href')

    file_name = file_url.split('/')[-1]

    r = requests.get(file_url)
    content = r.content
    buffer = pa.py_buffer(content)
    parquet_table = pq.read_table(buffer)
    pq.write_table(parquet_table, os.path.join(os.getcwd(), 'Data', 'Comstock', 'upgrade_agg', f'{file_name}'))
