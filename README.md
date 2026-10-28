# CTEMPR – sayısal deneyler

Makaledeki ("Conic Transformational Enhanced Multivariance Products Representation",
E. Şen, Z. Karabulut) tüm tablo ve şekilleri üreten Python kodu.

## Çalıştırma

    pip install -r requirements.txt
    python ctempr.py

Çalışma birkaç saniye sürer. Çıktılar betiğin bulunduğu klasöre yazılır:

- `figures/*.pdf`: makaledeki Şekil 1–6
- `results.json`: Tablo 1–4'teki ve bivariate örnekteki tüm sayılar

## Kodun yapısı

- `TEMPR(f, a, b, order, m, W, s)`: tek değişkenli dönüşümsel EMPR.
  `order=1` ATEMPR, `order=2` CTEMPR; `m` katsayı polinomlarının derecesi,
  `W` ağırlık fonksiyonu, `s` destek fonksiyonu (ikisi de içeride normalize edilir).
  Nesne `F(x)` biçiminde çağrılarak yaklaşım değerleri alınır; `sigma0` değişmezlik ölçenidir.
- `solve_constancy`: Önerme 1'deki özdeğer problemini SVD/QR ile kararlı biçimde çözer.
- `tempr2d`: iki değişkenli örnek (e^{x+y}, [0,1]^2).

Örnekler betiğin alt kısmında sırayla tanımlıdır; yeni bir fonksiyon denemek için
örneğin `TEMPR(np.sin, 0.1, 1, order=2).sigma0` yeterlidir.
