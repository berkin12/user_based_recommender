############################################
# User-Based Collaborative Filtering
#############################################
#user benzerlikleri üzerinden öneri yapıcaz şimdi
#kripto sinana alışkanlıkarı benzeyenlere bakıcaz

#BEĞENME ALIŞKANLIKLARINA BAK SANA BENZEYENLER BEĞENDİ SEN DE BEĞENİRSİN KARDEŞ

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df
#yukarıdaki işlemler yavaşlayabilir bilgisayarda istersen sabit kalma değerleri 10bin filan yap

user_movie_df = create_user_movie_df()

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#yukarıda random bi kullanıcı seçiyoruz çıktılarımız aynı olsu hocayla  random state 45 yaptık ke 
#bir de en dış parantezde çıktı string oluyor biz integer'a çevirdik 
#out:28941

#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user] #üst kısımda seçtiğimiz random user sinanı seçtik veri setimiz içerisinden
#yukarıdaki kodun çıktısı sadece sinan bazında seçme yaptı ve tüm filmleri çağırdı


movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#yukarıdaki kodla sinanın izlediği filmleri seçmeye çalışıyoruz listeden nasıl? sinanın dataframein sütunlarına git dedik notna ' mi diye sorduk dolu mu diye sorduk
#boş olmayanları getirecek şimdi bize


#şimdi yukarıda güzel yaptık ama bi kontrol edelim, aşağıda bakabiliriz vaziyet nedir diye index'e random userı koyduk çünkü satırlar kullanıcı
#sütuna yukarıdaki çıktıdan gelen listeden sinanın filmlerinden birini yazdık sütunda filmler var çünkü 
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]


len(movies_watched)



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

movies_watched_df = user_movie_df[movies_watched]
#sadece izlenen filmlere ilişkin bir bilgimiz var artık tüm filmlerden izlene filmleri seçtikke
#138k civarı kullanıcı ve 33 filmdeyiz artık
#yukarıdaki dataframe i filmler bazında indirgedik

user_movie_count = movies_watched_df.T.notnull().sum() # her bi kullanıcının kaç tane film izlediğini veren bi çıktı


user_movie_count = user_movie_count.reset_index() #bunu yapma sebebimiz şu yukarıdaki kodun çıktısında userıd'ler ındexti indexte çıkardık onları

user_movie_count.columns = ["userId", "movie_count"] #yukarıdaki çıktıyı daha cici hale getirdik

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False) 
#işte dedik ki user movie countun içindeki movie countları 20den büyük olanları azalan şekilde movie counta göre sırala

user_movie_count[user_movie_count["movie_count"] == 33].count() #sinanın izlediği 33 filmi izleyen kaç kullanıcı var say kardeşim dedik
#out 17 dedi bize 

#karar noktası 17 aynı film izleyen kullanıcı yeter diyebilirsen ya da 20'den büyük aynı filmi izlemişleri seçersin

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"] #işte 20'den büyük aynı izleyenleri seç dedik

#bu aşağıda işlemi programatik hale getirdikk
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

#                                          bu aşdk indexte userıd var
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], # ne dedik: movies watched datafremainde movies watchedun indexlerine git kardeş
                      random_user_df[movies_watched]])                                    #sinanın izlediğiyle aynı olan filmlere bi bak bakalım seç dyoruz
         # concat ile birleştiriyoruz şimdi bu içeridekileri neye göre random userımız sinana göre ve mowies watched izlediği filmlere göre
    
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


