import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.shape

# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.
# Adım 1: Ürünün ortalama puanını hesaplayınız.

df["overall"].mean()

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

#• reviewTime değişkenini tarih değişkeni olarak tanıtmanız
df.info()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

#• reviewTime'ın max değerini current_date olarak kabul etmeniz
current_date = pd.to_datetime('2014-12-07 00:00:00')

#• her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve gün cinsinden ifade edilen
#değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız
#gerekir. Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()


df["days"].quantile(0.25)
df["days"].quantile(0.50)
df["days"].quantile(0.75)
df["days"].describe().T

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= df["days"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.25)) & (dataframe["days"] <= df["days"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.50)) & (dataframe["days"] <= df["days"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

print(df.loc[df["days"] <= df["days"].quantile(0.25), "overall"].mean())
print(df.loc[(df["days"] > df["days"].quantile(0.25)) & (df["days"] <= df["days"].quantile(0.50)), "overall"].mean())
print(df.loc[(df["days"] > df["days"].quantile(0.50)) & (df["days"] <= df["days"].quantile(0.75)), "overall"].mean())
print(df.loc[(df["days"] > df["days"].quantile(0.75)), "overall"].mean())

#4.6957928802588995
#4.636140637775961
#4.571661237785016
#4.4462540716612375

# Burdaki ortalamaları gözlemlediğimizde; Son zamanlarda bu ürünün daha fazla tercih edildiğini görülmektedir. Ayrıca
# bu zaman aralığında ürün geliştiricilerin geçmiş günlere bakarak ürünle ilgili eski yorumlar özelinde geliştirmelere
# yaptığını gösterebilir.Diğer zamanlarda ki ortalamaların düşük olması ise o zamanlarda ürünün daha az tercih edildiği
# ya da ürünle ilgili memnuniyet durumunun daha az olduğunu gösterebilir. Reklam çalışmalarının yetersiz olduğunu
# gösterebilir.


# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# Adım 1: helpful_no değişkenini üretiniz.
#• total_vote bir yoruma verilen toplam up-down sayısıdır.
#• up, helpful demektir.
#• Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
#• Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df.info()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()
df["helpful_no"].value_counts()

df.sort_values("helpful", ascending=False).head()

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

#• score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
#score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
#• score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
#• score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
#• wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0: # burda giriceğimiz değerler sıfıra eşitse sıfır ver dedik.
        return 0
    return up / (up + down)

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


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
#• wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
#• Sonuçları yorumlayınız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Sonuçları inceleyecek olursak; en geçerli sonucun WLB'de göre alındığını görmekteyiz. Diğer skorları incelediğimizde
# basit düzeyde bir sonuç vermektedir. Fakat WLB dışındaki verilen bu skor ne kadar ikna edici veya bu sonuç ne kadar
# geçerli sorusunu akla getirmektedir. WLB aslında bu sorumuza daha geçerli sonuç vermektedir. Ör: score average rat değeri
# 1 olan bir yorumun WLB değerinin 0.65 olduğunu görmekteyiz. Sadece score average rat' e göre yorumlarsak bu sonuç çokta geçerli
# olmayabilir. Çünkü WLB'nin verdiği hassas skor aslında bu bilginin çok az insan tarafından değerlendirildiğini ve
# yüksek puan verdiğini göstermektedir. Sosyal açıdan sonucu değerlendirecek olursak eğer bir ürün ne kadar çok kişiye
# ulaşır ve o oranda tercih edilirse alınabilirliği vardır. WLB burda bu imkanı sağlamaktadır. Özetle amacımız:
# Topluluğun bilgeliğinde ölçümü hassaslaştırarak daha geçerli sonuçlar elde etmektir.
