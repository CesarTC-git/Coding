#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd

# Dates considered
Initial_day = '01' 
Initial_month = '01'
Final_day = '31'
Final_month = '12'
years = [str(i) for i in range(1969, 2021)]

# Webdriver options
options = webdriver.ChromeOptions() # Set up driver
options.add_argument('--disable-extensions')

reg_Madrid_value = '3195' # Considered values of second drop-down list (RETIRO)

# GET INTO THE WEBPAGE
for provincia in ['MADRID']: # Considering only MADRID
    for year in years:
        driver = webdriver.Chrome("Chromedriver.exe", chrome_options=options) # set up driver
        driver.minimize_window()
        time.sleep(1) # waiting 1 second before continuing
        
        driver.get('https://datosclima.es/Aemethistorico/Meteostation.php') # open the website
        
        # Find first drop-down menu and assign value
        select_prov = Select(driver.find_element(By.NAME, "Provincia"))
        select_prov.select_by_value(provincia)
        
        # Find second drop-down menu and assign value     
        WebDriverWait(driver,5).until(EC.element_to_be_clickable((By.NAME, "id_hija")))
        select_reg = Select(driver.find_element(By.NAME, "id_hija"))
        select_reg.select_by_value(reg_Madrid_value) # Assign value 'MADRID-RETIRO'
        
        # Find date fields
        element_Iday = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[2]/input[1]'
        element_Imonth = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[2]/input[2]'
        element_Iyear = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[2]/input[3]'
        element_Fday = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[5]/input[1]'
        element_Fmonth = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[5]/input[2]'
        element_Fyear = '//*[@id="col2"]/div[1]/div/form/div/table/tbody/tr/td[5]/input[3]'
        
        # Fill in date values
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Iday)))            .send_keys(Inicial_day)
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Imonth)))            .send_keys(Inicial_month)
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Iyear)))            .send_keys(year)
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Fday)))            .send_keys(Final_day)
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Fmonth)))            .send_keys(Final_month)
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_Fyear)))            .send_keys(year)

        # Press button 'BUSCAR'
        element_BUSCAR = '//*[@id="col2"]/div[1]/div/form/input[5]'
        WebDriverWait(driver,5)            .until(EC.element_to_be_clickable((By.XPATH, element_BUSCAR)))            .click()

        # Define XPATH for 3 tables of interest
        element_tabla_full1 = '/html/body/div[3]/div[2]/div/div[1]/table[2]/tbody/tr/td[1]/div/table'
        element_tabla_full2 = '/html/body/div[3]/div[2]/div/div[1]/table[2]/tbody/tr/td[2]/table'
        element_tabla_full3 = '/html/body/div[3]/div[2]/div/div[1]/table[2]/tbody/tr/td[3]/table'
        try: # Find 3 tables of interest
            WebDriverWait(driver,5)                .until(EC.element_to_be_clickable((By.XPATH, element_tabla_full1)))
            WebDriverWait(driver,5)                .until(EC.element_to_be_clickable((By.XPATH, element_tabla_full2)))
            WebDriverWait(driver,5)                .until(EC.element_to_be_clickable((By.XPATH, element_tabla_full3)))
        except: # If not found, raise an exception and continue with next iteration
            print('Empty data at {}'.format(provincia))
            driver.quit()
            continue

        # Assign tables to variables
        texto_tabla_full1 = driver.find_element(By.XPATH,element_tabla_full1)
        texto_tabla_full2 = driver.find_element(By.XPATH,element_tabla_full2)
        texto_tabla_full3 = driver.find_element(By.XPATH,element_tabla_full3)

        # Create a joint table with the 3 tables of interest
        tabla_full1 = list()
        tabla_full2 = list()
        tabla_full3 = list()
        for i in range(1, len(texto_tabla_full1.text.split('\n'))):
                       tabla_full1.append(texto_tabla_full1.text.split('\n')[i].split(' '))
        for i in range(1, len(texto_tabla_full2.text.split('\n'))):
                       tabla_full1.append(texto_tabla_full2.text.split('\n')[i].split(' '))
        for i in range(1, len(texto_tabla_full3.text.split('\n'))):
                       tabla_full1.append(texto_tabla_full3.text.split('\n')[i].split(' '))
        tabla_full = [tabla_full1, tabla_full2, tabla_full3]

        fecha = list()
        T_max = list()
        T_min = list()

        # Create 3 lists with the different columns in table_full
        for i in range(0, len(tabla_full[0])):
            fecha.append(tabla_full[0][i][0])
            T_max.append(tabla_full[0][i][1])
            T_min.append(tabla_full[0][i][2])

        # Create the DataFrame with the data and export to '.csv' file
        df_full = pd.DataFrame({'Fecha': fecha, 'MAX Temp': T_max, 'MIN Temp': T_min})
        df_full.to_csv('{}{}{}-{}{}{}_{}.csv'.format(Inicial_day, Inicial_month, year,                                                          Final_day, Final_month, year, provincia), index=False)
        #print(df_full)


        driver.quit() # Close windows

