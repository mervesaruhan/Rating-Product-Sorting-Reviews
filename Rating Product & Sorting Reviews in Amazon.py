
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df=pd.read_csv(r'C:\Users\SARU\Desktop\VBO\Measurement Problems\Case Study I\Rating Product&SortingReviewsinAmazon\amazon_review.csv')

df.info()
df.head()
df['asin'].value_counts()
df.shape

df['overall'].mean()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

df['reviewTime'].max()
df['reviewTime'].min()


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

##4.698716


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

df['helpful_no']=df['total_vote']-df['helpful_yes']

##df['helpful_no'].nunique()

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

df[df['helpful_yes'] > 0]['helpful_yes'].count()
df[df['helpful_no'] > 0]['helpful_no'].count()
df[df['total_vote'] > 0]['total_vote'].count()
df[df['total_vote'] == 0]['total_vote'].count()
df['total_vote'].count()
df['helpful_yes'].count()

#aynı ürüne hem yes hem no olanlar var



def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def score_up_down_diff(up, down):
    return up - down


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.info()
df.sort_values('wilson_lower_bound', ascending=False).head(20)
df['wilson_lower_bound'].dtypes






**************NOTLAR*********************



###BARS, her öğenin oylarını ve oy sayısını hesaba katarak bir puanlama yapar.
# Bu yöntemde, öğelerin puanlarına yapılan oylar arasındaki farklar göz önüne alınarak,
# daha az oylanan öğelere verilen puanların daha güvenilir olması sağlanır. Böylece,
# örneğin, 10.000 oy alan bir öğenin puanı, 100 oy alan bir öğenin puanından daha güvenilirdir.

##WLB ise, öğelerin puanlamalarını hesaplamak için bir alt sınır hesaplar. Bu alt sınır,
# bir öğenin aldığı oyların sayısına ve aldığı puanlara göre hesaplanır ve o öğenin gerçek puanlamasını
# tahmin etmek için kullanılır. WLB, daha az oy alan öğelerin puanlamasını düşük tutar ve daha fazla oy
# alan öğelerin puanlamasını yükseltir. Bu yöntem, daha az oy alan öğelerin haksız yere yüksek
# bir puan almasını engeller.

##BARS ve WLB arasındaki temel fark, hesaplama yöntemleridir. BARS, bütün oyları ve oy sayılarını hesaba
# katarak puanlama yaparken, WLB sadece alt sınır hesaplayarak puanlama yapar. Bu nedenle, BARS daha güvenilir
# bir sonuç verirken, WLB daha adaletli bir sonuç verir.

#BARS, Bayesian Average Rating Score'ın kısaltmasıdır.
#WLB, Wilson Lower Bound'ın kısaltmasıdır.

###Youtube'da Youtuber A ve Youtuber B'nin videoları var. Her ikisi de çok izleniyor, ama A'nın videoları
# daha çok beğeni alıyor. B'nin videoları ise daha çok beğeni alsalar da, daha çok beğenmeme de alıyorlar.BARS ve WLB
# skorları, bu duruma farklı yaklaşıyor. BARS, videonun izlenme sayısı ve beğeni sayısı arasındaki ilişkiyi göz önünde
# bulundurarak bir hesaplama yapıyor. Videonun beğeni sayısı yüksek ve izlenme sayısı da yüksekse, BARS skoru yüksek olacaktır.
# WLB ise, videonun kalitesini ölçmek için kullanıcıların beğenme ve beğenmeme sayısını dikkate alıyor. Bu hesaplamada
# sadece beğeni sayısını değil, beğenmeme sayısını da dikkate alıyor. Buna göre, B'nin videoları, daha fazla beğenmeye
# karşın daha çok beğenmeme aldığı için WLB skoru düşük olacaktır.Yani BARS, videonun popülerliğini gösterirken, WLB videonun kalitesini gösteriyor


