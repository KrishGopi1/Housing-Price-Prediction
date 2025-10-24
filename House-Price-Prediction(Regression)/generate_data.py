import pandas as pd
import random
import os

def create_dataset(path, n=5000, seed=0):
    random.seed(seed)
    rows = []
    neighborhoods = ['A','B','C','D','E']
    for i in range(n):
        sqft = int(max(300, random.gauss(1500, 500)))
        beds = random.choices([1,2,3,4,5],[0.05,0.25,0.4,0.2,0.1])[0]
        baths = max(1, int(round(beds - random.choice([0,0,1]) + random.choice([0,0,1]))))
        age = abs(int(random.expovariate(1/20)))
        lot = int(max(200, random.gauss(5000,2000)))
        garage = random.choice([0,1,2])
        neighborhood = random.choice(neighborhoods)
        dist_city = round(abs(random.gauss(10,6)),2)
        base = 50000
        price = base + sqft*120 + beds*8000 + baths*4000 + lot*2 - age*1000 - dist_city*500
        nb_mult = {'A':1.4,'B':1.15,'C':1.0,'D':0.9,'E':0.75}[neighborhood]
        price = int(price * nb_mult) + random.randint(-10000,10000)
        rows.append({
            'sqft':sqft,'beds':beds,'baths':baths,'age':age,'lot':lot,'garage':garage,
            'neighborhood':neighborhood,'dist_city':dist_city,'price':price
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    create_dataset('data/house_prices.csv', n=5000, seed=0)
    import pandas as pd
    df = pd.read_csv('data/house_prices.csv')
    sample = df.sample(10, random_state=1).drop(columns=['price'])
    sample.to_csv('data/sample_input.csv', index=False)
    print('data/house_prices.csv and data/sample_input.csv created')
