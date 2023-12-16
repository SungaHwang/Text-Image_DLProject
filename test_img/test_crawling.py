# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:08:38 2023

@author: Hydra18
"""
from selenium.webdriver.common.keys import Keys
import urllib.request
import time
import re
import requests
import csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from datetime import date, timedelta
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

my_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"

option = Options()
option.add_argument("start-maximized");
option.add_argument("disable-infobars")
option.add_argument("disable-extensions")
option.add_argument('disable-gpu')
#option.add_argument('headless')
option.add_argument('enable-automation')
option.add_argument('disable-dev-shm-usage')
option.add_argument('dns-prefetch-disable')
option.add_argument('no-sandbox')
option.add_argument('disable-browser-side-navigation')
option.add_argument(f"--user-agent={my_user_agent}")

dataset_folder = r"C:/Users/Andlab/jupyter_study/Text-Image_DLProject/test_img/"
#
"""
http://mss.kr/2702396
"""

url = 'https://www.musinsa.com/app/'
browser = webdriver.Chrome(options = option)
browser.get(url)

time.sleep(3)

# Login Page
browser.find_element(By.XPATH, '//*[@id="topCommonPc"]/header/div[3]/button').click()
time.sleep(3)

# ID / PW
browser.find_element(By.XPATH, '/html/body/div[1]/section[1]/div[1]/form/div[1]/div[1]/div/input').send_keys("id") # ID 입력
browser.find_element(By.XPATH, '/html/body/div[1]/section[1]/div[1]/form/div[1]/div[2]/div/input').send_keys("pw") # PW 입력
browser.find_element(By.XPATH, '/html/body/div[1]/section[1]/div[1]/form/div[2]/button').click()
time.sleep(3)


category = 'test'

data = f'{category}.txt'


# Open the file in read mode
def scrape_product_info_and_reviews(browser, data):
    with open(dataset_folder + data, 'r') as file:
        # Iterate over each line in the file
        for i, line in enumerate(file):
            print(i)
            browser.get(line)
            browser.implicitly_wait(2) # seconds
            time.sleep(2)
            
            #### Product Information Section #####
            # Part 1. #
            temporary = browser.page_source
            product_page = BeautifulSoup(temporary, 'html.parser')
            
            # Product Title (한글)
            n_prod_name_kor = '제품명'
            prod_name_kor = ''
            prod_name_kor = product_page.find_all('span', {'class' : 'product_title'})[0].find('em').text.replace(':', '.').split('/')[0].strip()
            
            # Product Info
            # Brand
            n_product_brand = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('p', {'class' : 'product_article_tit'})[0].text.split('/')[0].strip()
            product_brand = ''
            product_brand = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('p', {'class' : 'product_article_contents'})[0].text.split('/')[0].strip()
            
            # Prod Number
            # n_product_number = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('p', {'class' : 'product_article_tit'})[0].text.split('/')[1].strip()
            # product_number = ''
            # product_number = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('p', {'class' : 'product_article_contents'})[0].text.split('/')[1].strip()
            
            # Season
            #n_product_season = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('a', {'class' : 'ui-toggle-btn'})[0].text[:2]
            #product_season = ''
            #product_season = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('p', {'class' : 'product_article_contents'})[1].find('strong').text.replace('                          ', ' ').strip()
            
            # Gender
            # n_product_gender = "성별"
            # product_gender = ''
            # product_gender = product_page.find_all('ul', {'class' : 'product_article'})[0].find_all('span', {'class' : 'txt_gender'})[0].text.replace('\n', '')
            
            # # Search (1st Month)
            # n_product_search_1st_month = product_page.find_all('p', {'class' : 'product_article_tit'})[2].text
            # product_search_1st_month = ''
            # product_search_1st_month = product_page.find_all('p', {'class' : 'product_article_contents'})[2].find('strong').text
            
            # # Accum Search (Last Year)
            # n_product_search_last_year = product_page.find_all('p', {'class' : 'product_article_tit'})[3].text
            # product_search_last_year = ''
            # product_search_last_year = product_page.find_all('p', {'class' : 'product_article_contents'})[3].find('strong').text
            
            # # Likes
            # n_product_likes = product_page.find_all('p', {'class' : 'product_article_tit'})[4].text
            # product_likes = ''
            # product_likes = product_page.find_all('p', {'class' : 'product_article_contents'})[4].find('span').text
            
            # #### Review Section #####
            # # <-- New Page --> #
            try:
                # browser.find_element(By.XPATH, '//*[@id="reviewGoodsSimilarList"]/option[1]').click()
                browser.find_element(By.XPATH, '//*[@id="reviewSelectSort"]').click()
                temporary = browser.page_source
                product_review_sec = BeautifulSoup(temporary, 'html.parser')
                
            #     # Total Number of Purchased Reviews
            #     n_purchase_reviews = product_review_sec.find_all('h3', {'class' : 'title-box font-mss'})[0].text[:4]
            #     purchase_reviews = ''
            #     purchase_reviews = product_review_sec.find_all('h3', {'class' : 'title-box font-mss'})[0].text[4:].replace(')','').replace('(','')

                # product_info = [{n_prod_name_kor  : prod_name_kor,   # 제품명
                #                  n_product_brand  : product_brand,   # 브랜드명
                #                 #  n_product_number : product_number,  # 제품번호
                #                  #n_product_season : product_season,  # 계절
                #                  n_product_gender : product_gender,  # 성별
                #                  n_product_search_1st_month : product_search_1st_month,
                #                  n_product_search_last_year : product_search_last_year,
                #                  n_product_likes  : product_likes,  # 좋아요 수
                #                  n_purchase_reviews : purchase_reviews}] # 후기 수
                
                # # Convert the list of dictionaries to a dataframe
                # product_info_df = pd.DataFrame(product_info)
                
                Path(dataset_folder + category+'_info/'+ prod_name_kor).mkdir(parents = True, exist_ok = True)
                # product_info_df.to_csv(dataset_folder +category+'_info/'+ prod_name_kor + '/' + line.split('/')[-1].replace('\n', '') + ".csv", index = False, encoding = 'utf-8-sig')
                
                print(prod_name_kor)
                print()
                #############################
                # Part 2. #
                with open(dataset_folder + category+'_info/'+ prod_name_kor + '/' + 'reviews.csv', 'w', encoding='utf-8-sig', newline = '') as file:    
                    csvfile = csv.writer(file)
                    reviews = []
                    for k in range(1, 10): # 수정 (페이지)
                        # print("Page Number: {}".format(k))
                        if k > 500: 
                            break
                        else:
                            outer_lists = []
                            for j in range(3,9): ### ??? 3, 9
                                browser.find_element(By.XPATH, '//*[@id="reviewListFragment"]/div[11]/div[2]/div/a[' + str(j) + ']').send_keys(Keys.ENTER) #.click()
                                time.sleep(1)
                                
                                review_site = browser.page_source
                                product_review_page = BeautifulSoup(review_site, 'html.parser')
                                
                                temp_list = []
                                # Temp_list Lists (Max. 10 per page)
                                for i in range(len(product_review_page.find_all('div', 'review-list'))):
                        
                                    # Profile name
                                    n_profile_name = ''
                                    n_profile_name = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', {'class' : 'review-profile__name'})[0].text
                                    
                                    try:
                                        # Profile Gender
                                        n_profile_gender = ''
                                        n_profile_gender = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[0].strip()
                                        
                                        # Profile Height
                                        n_profile_height = ''
                                        n_profile_height = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[1].strip()
                                        
                                        # Profile Weight
                                        n_profile_weight = ''
                                        n_profile_weight = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[2].strip()
                                        
                                    except:
                                        n_profile_gender = ''
                                        n_profile_height = ''
                                        n_profile_weight = ''
                                    
                                    # Profile Purchased Product Name
                                    profile_purchased_prod_name = ''
                                    profile_purchased_prod_name = product_review_page.find_all('div', {'class' : 'review-goods-information'})[i].find_all('div', {'class': 'review-goods-information__item'})[0].find('a').text
                                    
                                    # Profile Purchased Product Unique Code
                                    profile_purchased_product_unique_code = ''
                                    profile_purchased_product_unique_code = product_review_page.find_all('div' , {'class' : 'review-contents'})[i]['data-goods-no']
                            
                                    # Profile Purchased Product Size
                                    profile_purchased_prod_size = ''
                                    profile_purchased_prod_size = product_review_page.find_all('div', {'class' : 'review-goods-information'})[i].find_all('div', {'class': 'review-goods-information__item'})[0].find('span').text.replace('\n','').replace(', null','').strip()
                        
                                    # Profile Purchased Product Ratings
                                    profile_purchased_prod_ratings = ''
                                    profile_purchased_prod_ratings = product_review_page.find_all('div', {'class' : 'review-list__rating-wrap'})[i].find_all('span', {'class' : 'review-list__rating__active'})[0]['style'][-5:].replace(':', '').strip()
                        
                                    # Profile Purchased Comments
                                    profile_purchased_comments = ''
                                    profile_purchased_comments = product_review_page.find_all('div' , {'class' : 'review-contents'})[i].find_all('div', {'class' : 'review-contents__text'})[0].text
                                    
                                    # Profile Purchased Comments Unique Code
                                    profile_purchased_comment_unique_code = ''
                                    profile_purchased_comment_unique_code = product_review_page.find_all('div' , {'class' : 'review-contents'})[i]['data-review-no']
                                    
                                    # Profile Purchased Photo
                                    # creating a new directory based on the profile_purchased_product_unique_code
                                    Path(dataset_folder + category).mkdir(parents = True, exist_ok = True) # 수정 prod_name_kor + 
                                    # Path(dataset_folder + prod_name_kor + '/'+ profile_purchased_product_unique_code)

                                    try:
                                        profile_purchased_prod_photo = ''    
                                        profile_purchased_prod_photo = 'https://' + product_review_page.find_all('div' , {'class' : 'review-content-photo'})[i].find('img')['src'].replace('//', '').replace('.view', '')
                                        urllib.request.urlretrieve(profile_purchased_prod_photo, dataset_folder + category +'/' + profile_purchased_comment_unique_code + ".jpg")
                                        time.sleep(2)
                                        
                                    except:
                                        profile_purchased_prod_photo = ''    
                                
                                    # profile directory
                                    profile_directory = ''
                                    profile_directory = profile_purchased_prod_photo, dataset_folder + category + '/' + profile_purchased_comment_unique_code + ".jpg"
                        
                                    temp_list.append([n_profile_name, n_profile_gender, n_profile_height, n_profile_weight, 
                                                      profile_purchased_prod_name, profile_purchased_product_unique_code, 
                                                      profile_purchased_prod_size, profile_purchased_prod_ratings, 
                                                      profile_purchased_comments, profile_purchased_comment_unique_code, profile_directory])
                                
                                outer_lists.extend(temp_list)
                                
                            for row in outer_lists:
                                csvfile.writerow(row)
            except:
                continue
                
                # temporary = browser.page_source
                # product_review_sec = BeautifulSoup(temporary, 'html.parser')
                
                # # Total Number of Purchased Reviews
                # # n_purchase_reviews = product_review_sec.find_all('h3', {'class' : 'title-box font-mss'})[0].text[:4]
                # # purchase_reviews = ''
                # # purchase_reviews = product_review_sec.find_all('h3', {'class' : 'title-box font-mss'})[0].text[4:].replace(')','').replace('(','')
    
                # # product_info = [{n_prod_name_kor  : prod_name_kor,   # 제품명
                # #                  n_product_brand  : product_brand,   # 브랜드명
                # #                 #  n_product_number : product_number,  # 제품번호
                # #                  #n_product_season : product_season,  # 계절
                # #                 #  n_product_gender : product_gender,  # 성별
                # #                 #  n_product_search_1st_month : product_search_1st_month,
                # #                 #  n_product_search_last_year : product_search_last_year,
                # #                 #  n_product_likes  : product_likes,  # 좋아요 수
                # #                  n_purchase_reviews : purchase_reviews}] # 후기 수
                
                # # # Convert the list of dictionaries to a dataframe
                # # product_info_df = pd.DataFrame(product_info)
                
                # Path(dataset_folder + category+'_info/'+ prod_name_kor).mkdir(parents = True, exist_ok = True)
                # # product_info_df.to_csv(dataset_folder + category+'_info/'+ prod_name_kor + '/' + line.split('/')[-1].replace('\n', '') + ".csv", index = False, encoding = 'utf-8-sig')
                
                # print(prod_name_kor)
                # print()
                # #############################
                # # Part 2. #
                # with open(dataset_folder +category+ '_info/'+ prod_name_kor + '/' + 'reviews.csv', 'w', encoding='utf-8-sig', newline = '') as file:    
                #     csvfile = csv.writer(file)
                #     reviews = []
                #     for k in range(1, 2): # 수정 int(product_review_sec.find_all('div', {'class' : 'box_page_msg'})[0].text.replace('\n', '').strip().split(' ')[0]), 5
                #         print("Page Number: {}".format(k))
                #         if k > 1000: 
                #             break
                #         else:
                #             outer_lists = []
                #             for j in range(3, 5): ### 3, 9 ???
                #                 browser.find_element(By.XPATH, '//*[@id="reviewListFragment"]/div[11]/div[2]/div/a[' + str(j) + ']').send_keys(Keys.ENTER) # .click()
                #                 time.sleep(1)
                                
                #                 review_site = browser.page_source
                #                 product_review_page = BeautifulSoup(review_site, 'html.parser')
                                
                #                 temp_list = []
                #                 # Temp_list Lists (Max. 10 per page)
                #                 for i in range(len(product_review_page.find_all('div', 'review-list'))):
                        
                #                     # Profile name
                #                     n_profile_name = ''
                #                     n_profile_name = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', {'class' : 'review-profile__name'})[0].text
                                    
                #                     try:
                #                         # Profile Gender
                #                         n_profile_gender = ''
                #                         n_profile_gender = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[0].strip()
                                        
                #                         # Profile Height
                #                         n_profile_height = ''
                #                         n_profile_height = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[1].strip()
                                        
                #                         # Profile Weight
                #                         n_profile_weight = ''
                #                         n_profile_weight = product_review_page.find_all('div', 'review-list-wrap')[0].find_all('div', {'class' : 'review-list'})[i].find_all('p', 'review-profile__body_information')[0].text.split('·')[2].strip()
                                        
                #                     except:
                #                         n_profile_gender = ''
                #                         n_profile_height = ''
                #                         n_profile_weight = ''
                                    
                #                     # Profile Purchased Product Name
                #                     profile_purchased_prod_name = ''
                #                     profile_purchased_prod_name = product_review_page.find_all('div', {'class' : 'review-goods-information'})[i].find_all('div', {'class': 'review-goods-information__item'})[0].find('a').text
                                    
                #                     # Profile Purchased Product Unique Code
                #                     profile_purchased_product_unique_code = ''
                #                     profile_purchased_product_unique_code = product_review_page.find_all('div' , {'class' : 'review-contents'})[i]['data-goods-no']
                                    
                #                     # Profile Purchased Product Size
                #                     profile_purchased_prod_size = ''
                #                     profile_purchased_prod_size = product_review_page.find_all('div', {'class' : 'review-goods-information'})[i].find_all('div', {'class': 'review-goods-information__item'})[0].find('span').text.replace('\n','').replace(', null','').strip()
                        
                #                     # Profile Purchased Product Ratings
                #                     profile_purchased_prod_ratings = ''
                #                     profile_purchased_prod_ratings = product_review_page.find_all('div', {'class' : 'review-list__rating-wrap'})[i].find_all('span', {'class' : 'review-list__rating__active'})[0]['style'][-5:].replace(':', '').strip()
                        
                #                     # Profile Purchased Comments
                #                     profile_purchased_comments = ''
                #                     profile_purchased_comments = product_review_page.find_all('div' , {'class' : 'review-contents'})[i].find_all('div', {'class' : 'review-contents__text'})[0].text
                                    
                #                     # Profile Purchased Comments Unique Code
                #                     profile_purchased_comment_unique_code = ''
                #                     profile_purchased_comment_unique_code = product_review_page.find_all('div' , {'class' : 'review-contents'})[i]['data-review-no']
                                    
                #                     # Profile Purchased Photo
                #                     # creating a new directory based on the profile_purchased_product_unique_code
                #                     Path(dataset_folder + category+'/').mkdir(parents = True, exist_ok = True)
                                    
                #                     try:
                #                         profile_purchased_prod_photo = ''    
                #                         profile_purchased_prod_photo = 'https://' + product_review_page.find_all('div' , {'class' : 'review-content-photo'})[i].find('img')['src'].replace('//', '').replace('.view', '')
                #                         urllib.request.urlretrieve(profile_purchased_prod_photo, profile_purchased_prod_photo, dataset_folder + category+'/' + profile_purchased_comment_unique_code + ".jpg")
                #                         time.sleep(2)
                #                     except:
                #                         profile_purchased_prod_photo = ''
                                    
                #                     # profile directory
                #                     profile_directory = ''
                #                     profile_directory = profile_purchased_prod_photo, dataset_folder + category+'/' + profile_purchased_comment_unique_code + ".jpg"
                        
                #                     temp_list.append([n_profile_name, 
                #                                       n_profile_gender,
                #                                       n_profile_height, 
                #                                       n_profile_weight, 
                #                                       profile_purchased_prod_name,
                #                                       profile_purchased_product_unique_code, 
                #                                       profile_purchased_prod_size, 
                #                                       profile_purchased_prod_ratings, 
                #                                       profile_purchased_comments, 
                #                                       profile_purchased_comment_unique_code, 
                #                                       profile_directory])
                                
                                    
                        
                #                 outer_lists.extend(temp_list)
                                
                #             for row in outer_lists:
                #                 csvfile.writerow(row)

scrape_product_info_and_reviews(browser, data)