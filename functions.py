import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

#This module aims to automate a segmentation dataset givena period, and calculate ARI scores. 

#import datasets
customers = pd.read_csv('olist_customers_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')

#To build our functions we have to define new datasets

#number of different payment type by order
payments.set_index('order_id', inplace=True, drop=True)
orders['order_purchase_timestamp'] = pd.to_datetime(
                                            orders['order_purchase_timestamp']
                                            )

orders['order_delivered_customer_date'] = pd.to_datetime(
                                            orders['order_delivered_customer_date']
                                                        )
payment_types = pd.get_dummies(payments[['payment_type']])
payment_types = payment_types.groupby('order_id').sum()

# price payed by orders
price_payed_orders=pd.DataFrame(payments.groupby('order_id')['payment_value'].sum())

# number of items by order
nb_items=pd.DataFrame(items.groupby('order_id').agg({'product_id':'count'}))
nb_items.rename(columns={'product_id':'nb_items'},inplace=True)



def category(x):
    '''This function simplifies the category names'''
    if x in ['perfumaria',
             'beleza_saude',
             'fashion_bolsas_e_acessorios',
             'fashion_underwear_e_moda_praia',
             'fashion_roupa_masculina', 
             'fashion_calcados',
             'fashion_roupa_feminina',
             'fashion_esporte',
             'relogios_presentes',
             'fashion_calcados']:
        return 'fashion_beauty'
    elif x in ['cama_mesa_banho',
               'moveis_escritorio',
               'moveis_sala', 
               'moveis_colchao_e_estofado',
               'moveis_quarto',
               'moveis_decoracao',
               'casa_conforto',
               'casa_conforto_2',
               'portateis_cozinha_e_preparadores_de_alimentos',
               'moveis_cozinha_area_de_servico_jantar_e_jardim',
               'la_cuisine',
               'artigos_de_natal',
               'flores']:
        return 'furnitures_deco'
    elif x in ['telefonia',
               'eletronicos',
               'informatica_acessorios',
               'audio',
               'eletrodomesticos',
               'telefonia_fixa',
               'portateis_casa_forno_e_cafe',
               'eletroportateis',
               'tablets_impressao_imagem',
               'pc_gamer',
               'eletrodomesticos_2',
               'pcs']:
        return 'electronic'
    elif x in ['livros_tecnicos',
               'esporte_lazer',
               'consoles_games',
               'papelaria',
               'instrumentos_musicais',
               'livros_interesse_geral',
               'artes',
               'livros_importados',
               'cds_dvds_musicais',
               'artes_e_artesanato',
               "dvds_blu_ray",
               'cine_foto','musica']:
        return 'hobbies'
    elif x in ['alimentos','alimentos_bebidas','bebidas']:
        return 'food'
    elif x in ['ferramentas_jardim',
               'construcao_ferramentas_construcao',
               'construcao_ferramentas_iluminacao',
               'climatizacao','sinalizacao_e_seguranca',
               'construcao_ferramentas_jardim',
               'casa_construcao',
               'construcao_ferramentas_seguranca',
               'construcao_ferramentas_ferramentas'] : 
        return 'diy'
    elif x in ['automotivo']:
        return 'auto'
    elif x in ['bebes',
               'brinquedos',
               'fashion_roupa_infanto_juvenil']:
        return 'kids'
    elif x in ['cool_stuff',
               'pet_shop',
               'malas_acessorios', 
               'utilidades_domesticas',
               'fraldas_higiene',
               'artigos_de_festas']:
        return 'accessories'
    elif x in ['seguros_e_servicos',
               'agro_industria_e_comercio', 
               'industria_comercio_e_negocios']: 
        return 'services'
    else:
        return 'other'
    
    
    
#List of items by order with their dimensions
products['dimensions'] = products['product_length_cm']\
                        * products['product_height_cm']\
                        * products['product_width_cm']

products_orders = pd.merge(items[['product_id',
                                  'order_id']],
                           products[['product_id',
                                     'product_category_name',
                                     'dimensions']],
                           on='product_id',
                           how='inner')

#List of items with simplified category
products_orders['category'] = products_orders['product_category_name']\
                                             .apply(category)

products_orders.set_index('order_id', inplace=True)

products_orders.sort_values('order_id', inplace=True)

# we appy a get dummies to the simplified products categories
products_orders = pd.get_dummies(products_orders[['category',
                                                'dimensions']])


# List of orders with the number of products by category and the average dimension of its items

products_orders = products_orders.groupby('order_id')\
                              .agg({'dimensions': 'mean',
                                    'category_other': 'sum',
                                    'category_kids': 'sum',
                                    'category_hobbies': 'sum',
                                    'category_furnitures_deco': 'sum',
                                    'category_food': 'sum',
                                    'category_fashion_beauty': 'sum',
                                    'category_electronic': 'sum',
                                    'category_diy': 'sum',
                                    'category_auto': 'sum',
                                    'category_accessories': 'sum',
                                    'category_services': 'sum',
                                    'category_other': 'sum'
                                      })






# List of orders with the average review scores of its items
score_orders = pd.DataFrame(reviews.groupby('order_id')['review_score'].mean())

def create_dataset(first_date,last_date):
    ''' This function returns the segmentation dataset for orders delivered
    between first_date and last_date'''
    
    #Orders delivered between first_date and last_date
    orders_d = orders[(orders['order_purchase_timestamp'] > first_date)
                      & (orders['order_purchase_timestamp'] < last_date)
                      & (orders['order_status'] == 'delivered')]
    
    #Merge between customers and orders informations
    df = pd.merge(customers[['customer_id',
                             'customer_unique_id',
                             'customer_city']],
                  orders_d[['order_id',
                            'customer_id',
                            'order_purchase_timestamp',
                            'order_delivered_customer_date']],
                  on='customer_id',
                  how='inner')

    
    
    #Transform to date format 'order_purchase_timestamp' and 'order_delivered_customer_date'
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp']) 
    
    df['order_delivered_customer_date'] = pd.to_datetime(
                                            df['order_delivered_customer_date']
                                                        )

    #we calculate the number of days since orders
    lastdate = df['order_purchase_timestamp'].max()
    lastdate += datetime.timedelta(days=1)
    df['nb_days_since_order'] = (lastdate-df['order_purchase_timestamp']).dt.days
    
    

    #we define delivery number of days
    df['delivery_nb_of_days'] = (df['order_delivered_customer_date']
                                 - df['order_purchase_timestamp']).dt.days
    
    #We merge with payment types
    df = pd.merge(payment_types,
                  df,
                  how='inner',
                  on='order_id')
    
    # Number of payments by order
    df['nb_payments'] = df['payment_type_boleto']\
                        + df['payment_type_credit_card']\
                        + df['payment_type_debit_card']\
                        + df['payment_type_voucher']
    
    # Merge with orders' price
    df = pd.merge(df, price_payed_orders, on='order_id', how='inner')
    
    # Merge with number of items by order
    df = pd.merge(df, nb_items, how='left', on='order_id')
    
    # Merge with products categories and average dimensions of items 
    df = pd.merge(df, products_orders, on='order_id', how='left')
    
    #Merge with average scopre review by order
    df = pd.merge(df, score_orders, on='order_id', how='left')
    
    #Definition of city density
    nb_customers = len(df.customer_unique_id.unique())
    city_density = pd.DataFrame(100*df.groupby('customer_city')['customer_unique_id']
                                    .count()
                                / nb_customers)
    city_density.rename(columns={'customer_unique_id': 'city_density'}, inplace=True)
    
    #Merge to df
    df = pd.merge(df, city_density, how='left', on='customer_city')
    
       
    #aggregating data by customer
    dg_init = df.groupby('customer_unique_id').agg({#For each customer, we calculate
                                         #Number of orders
                                         'order_id':'count', 
                                            
                                         #the average nb of uses of each payment type
                                         'payment_type_boleto':'mean',
                                         'payment_type_debit_card':'mean',
                                         'payment_type_credit_card':'mean',
                                         'payment_type_voucher':'mean',
                                            
                                          #  the average number of payments
                                         'nb_payments':'mean',
                                            
                                          # the number of days since last order
                                         'nb_days_since_order':'min',
                                         
                                         #the average delivery number of days
                                         'delivery_nb_of_days':'mean',
                                            
                                         #the average payment value
                                         'payment_value':'mean',
                                            
                                         #the average number of items 
                                         'nb_items':'mean',
                                            
                                         #the average review score
                                         'review_score':'mean',
                                            
                                         # the customer's city density 
                                         'city_density':lambda x: x.max(), 
                                         
                                         # the average number of items for each category
                                         'category_other':'mean',
                                         'category_kids':'mean',
                                         'category_hobbies':'mean',
                                         'category_furnitures_deco':'mean',
                                         'category_food':'mean',
                                         'category_fashion_beauty':'mean',
                                         'category_electronic':'mean',
                                         'category_diy':'mean',
                                         'category_auto':'mean',
                                         'category_services':'mean',
                                         'category_accessories':'mean',
                                          
                                          #the average dimension of items
                                         'dimensions':'mean'
                                          })
    # we rename two columns
    dg_init.rename(columns={'order_id': 'nb_of_orders',
                            'nb_days_since_order': 'nb_days_since_last_order'},
                   inplace=True)
    
    dg=dg_init[['nb_of_orders',
                'payment_type_boleto',
                'nb_days_since_last_order',
                'delivery_nb_of_days',
                'payment_value',
                'review_score',        
                'city_density',
                'dimensions']]
    
    return dg_init, dg



def cluster(dg,n_cluster=7):
    '''This function applies the Kmeans algorithm on the segmentation dataset dg'''
    from sklearn import preprocessing
    from sklearn import cluster, metrics
    
    dg=dg.dropna()
    X=dg.values
    
    #standardize data
    std_sc=preprocessing.StandardScaler()
    X_norm=std_sc.fit_transform(X)
    
    #drop observations where payment value's z-score is >5
    dg=dg[(np.abs(X_norm[:,4])<5)]
    X_norm=X_norm[(np.abs(X_norm[:,4])<5)]
    
    #clustering
    cls=cluster.KMeans(n_clusters=7)
    cls.fit(X_norm)
    
    return dg, cls.labels_


len('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')   

def stability():
    '''returns the adjusted rand score between segmentation during the first year 
       and new segmentations made every months after the first year'''
    #u will contain the ARI scores
    u = [1]
    #n and cpt will contain the number of the month
    n = [0]
    cpt = 0
    
    first_date = datetime.datetime(2016,9,4)-datetime.timedelta(days=1)
    last_date = first_date+relativedelta(years=+1)

    #create first segmentation
    dg_init, dg = create_dataset(first_date,last_date)
    dg, labels_init = cluster(dg,n_cluster=7)
    dg["num_cluster"] = labels_init
    
    
    #create new segmentations each month after the first year
    while last_date < datetime.datetime(2018,10,17)+relativedelta(months=+1) :   
        cpt += 1
        n.append(cpt)
        
        last_date += relativedelta(months=+1)
        dg_init,dg2 = create_dataset(first_date,last_date)
        dg2,labels = cluster(dg2,n_cluster=7)
        
        dg2['num_cluster'] = labels
        
        #merge on customers that are in the initial segmentation
        dg2 = pd.merge(dg['num_cluster'],
                       dg2['num_cluster'],
                       on='customer_unique_id',
                       how='inner')
        
        from sklearn.metrics.cluster import adjusted_rand_score
        
        #print(adjusted_rand_score(dg2['num_cluster_x'],dg2['num_cluster_y']))
        u.append(adjusted_rand_score(dg2['num_cluster_x'],dg2['num_cluster_y']))
    
    return n,u




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    