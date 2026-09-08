# med-rag — Kullanım Kılavuzu

> Tek sayfalık özet. Kurulum/operasyon için [`DEPLOY.md`](DEPLOY.md),
> yapılandırma için [`CONFIG.md`](CONFIG.md).

## Giriş

Tarayıcıdan `http://<sunucu-adresi>:8507` açın. Şifrenizi girin
(`MEDRAG_SIFRE`). Oturum açık kaldıkça tekrar sormaz.

## Kütüphane

- **Belge ekleme**: Sol üstteki yükleme alanına dosyaları sürükleyin ya da
  tıklayıp seçin. Aynı anda birçok dosya seçilebilir. Desteklenen
  biçimler: PDF (dijital + taranmış/foto), Word, PowerPoint, Excel, HTML,
  Markdown.
- **Durum rozetleri**: Her dosya `kuyrukta → işleniyor → hazır` (ya da
  `hata`) rozetiyle izlenir. Taranmış PDF'ler yavaştır; sayfa sayfa
  ilerler. Hata olursa rozetin üstünde nedeni görürsünüz.
- **Belge okuma**: Listede bir belgeye tıklayın — düzenlenmiş, okunabilir
  hâli açılır (tablo/liste/başlıklar korunur). "Orijinali gör" ile kaynak
  PDF'i açarsınız.
- **Arama**: Listeyi dosya adına göre süzün.
- **Silme**: Çöp kutusu simgesi → onaylayın. Belgenin tüm izleri (metin,
  vektör, kaynak listesi) anında kaldırılır; sohbet geçmişindeki eski
  cevaplar belgeye artık atıf veremez.
- **Değiştirme**: Aynı ada sahip yeni sürümü yükleyin — eski içerik
  düşer, yenisi işlenir.

## Sohbet

- Sorunuzu yazın; cevap **akarken** hangi aşamadan geçtiği sağ panelde
  görünür.
- Cevaplardaki **[1] [2]** rozetleri kanıttır: tıklayınca ilgili belge,
  sayfa ve bölüme gidersiniz. Cevabın altında numaralı kaynak listesi olur.
- Kaynak bulunamazsa sistem **"bulamadım"** der; uydurmaz. Tıbbi nitelikli
  cevapların altında kısa bir **"hekime danışın"** notu çıkar.
- Çelişen kaynaklar varsa ikisi de gösterilir ve uyarı kutusu görünür.
- Sağ üstten **Yeni sohbet** başlatabilirsiniz; eski sohbetler geçmişten
  geri yüklenebilir.

## Notlar

- Notlar sekmesinden metin notu oluşturun/düzenleyin (ör. hasta özeti,
  hatırlatma).
- Kaydettiğiniz not otomatik işlenir ve chatbot **notunuzu da kaynak olarak
  kullanır** — cevaplarda notunuz atıf olarak görünür.
- Not silmek, belge silmekle aynıdır: izleri kaldırılır.

## Sık sorulanlar

**Belgem uzun süre "işleniyor"da kaldı?** Taranmış/foto PDF'ler normal
koşuda da dakikalar sürer. Çok gecikirse hata rozeti düşer; dosyayı yeniden
yüklemek (aynı adla) yeniden işlemeye yeter.

**Cevap eski içeriğe atıf veriyor?** Dosyayı değiştirdiyseniz aynı adla
yeniden yükleyin; eski türevler otomatik silinir.

**Şifremi unuttum?** Sunucudaki `.env` dosyasında `MEDRAG_SIFRE` satırını
güncelleyin ve `docker compose up -d web` ile web servisini yeniden
başlatın.
