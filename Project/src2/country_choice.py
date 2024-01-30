import pandas as pd
import os
names = ('Russia', 'China', 'Argentina', 'Brazil', 'Spain', 'France')
vals = (15508, 9333, 7092, 9201, 6340, 9406)
data = {'country': names, 'vals': vals}
df = pd.DataFrame(data)

def country_select():
    country = input('Select 5k country: ')
    if country in names:
        selected_vals = df.loc[df['country'] == country, 'vals'].values
        print(f'{selected_vals[0]} Tx in total')
        path=f'./database/{country}' + ".csv"
        path2=f'/home/pratush/5kRasters/{country}' +'_raster.tif'
        return selected_vals[0],path,path2
        
    else:
        print('Country not found in the list.')
        return None


	
