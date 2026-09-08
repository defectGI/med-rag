# med-rag — Geliştirme Planı

> Durum: UYGULAMA TAMAMLANDI (Grup A, B, C, D + E; F kabul aşamasında).
> Tam suite baseline'da: 2105 passed / 61 önceden-var-olan fail (fork
> baseline'ı) / sıfır regresyon; frontend `npm run build` yeşil; ruff yeni
> kodda temiz. B5 (sohbet geçmişi + yeni sohbet) tamamlandı. E grubu:
> compose (web + pipeline-worker + pipeline + qdrant), okuma-yazma volume
> şeması, gecelik yedek + geri yükleme prosedürü (`DEPLOY.md` §3.1),
> gözlemlenebilirlik ve auth/upload sertleştirmesi tamam. F: uçtan uca
> duman testi `tools/e2e_smoke.py` hazır; canlı koşum ilk deploy sonrası
> yapılacak. Kullanım kılavuzu: `KULLANIM.md`.


---

## 1. Vizyon

Ablanın (hekim) yükleyeceği tıbbi belgelerden — dijital PDF, taranmış/foto PDF
ve diğer ofis formatları — **kanıt (evidence) göstererek** cevap veren, kendi
kendine büyüyen bir doküman asistanı. Kullanıcı dosya ekledikçe sistem büyür,
dosya silindiğinde türevleri temizlenir, dosya değiştiğinde eski bilgi düşüp
yenisi işlenir. Arayüz Wada Sanzo paletinden fildişi + güneş tonlarıyla,
shadcn/ui üzerine kurulu modern bir SPA'dır.

### Kapsam dışı (non-goals)

- WhatsApp / harici mesajlaşma entegrasyonu (fork'ta zaten söküldü)
- Çok kullanıcılılık, rol yönetimi
- facts/specs.db + text2sql yolunun aktifleştirilmesi (kod duruyor, kullanılmıyor)
- Mobil native uygulama (responsive web yeterli)

## 2. Alınmış Kararlar (soruların cevapları)

| # | Konu | Karar |
|---|------|-------|
| K1 | Kurulum | Kullanıcının evindeki makinede, **Coolify** ile deploy |
| K2 | Dosya girişi | **UI'dan yükleme** (sürükle-bırak / çoklu seçim); izlenen klasör YOK |
| K3 | Kullanıcı | Tek kullanıcı + **basit şifre** (login ekranı, oturum çerezi) |
| K4 | Dil | **Tam karışık** TR/EN belge + soru; cevap sorunun dilinde |
| K5 | Ölçek | **Büyük: 500+ belge** — artımlılık, maliyet kontrolü ve durum takibi kritik |
| K6 | Dosya tipleri | Parser'ın desteklediği **tümü** (PDF, DOCX, PPTX, XLSX, HTML, MD) |
| K7 | LLM | Mevcut **sağlayıcı-bağımsız mimari** korunur; kullanımda **bulut modelleri** |
| K8 | Ön yüz | **Vite + React + TypeScript + Tailwind + shadcn/ui**; backend API Flask kalır |
| K9 | Tema | **Açık + koyu** tema toggle'ı; CSS değişkeni tabanlı |
| K10 | Palet | Wada Sanzo: **fildişi zemin + güneş tonları** (sarı/turuncu/kızıl vurgular) |
| K11 | UI kapsamı | **Kütüphane + chat + notlar + belge gezinme** (aşağıda detay) |
| K12 | Notlar | Kullanıcının oluşturduğu/düzenlediği **txt notlar da korpusa girer**, chatbot onları kaynak olarak kullanır |
| K13 | Belge inceleme | **Düzenlenmiş, kullanıcı-dostu Markdown render** (ham değil; Claude-artifacts hissi) + "orijinali gör" ile PDF |
| K14 | Evidence | **Hem satır içi rozetler hem toplanmış kaynak listesi**; atıf zorunlu |
| K15 | Çelişki/güvenlik | Çelişen kaynaklar **ikisi de + uyarı**; bulunamadıysa **"bulamadım"**; tıbbi cevapların altında kısa **"hekime danışın"** notu |
| K16 | İşleme durumu | Kütüphanede dosya başına **durum rozeti** (kuyrukta → işleniyor → hazır/hata), sayfa bazlı ilerleme |
| K17 | Yedekleme | **Gecelik otomatik**: korpus + notlar + Qdrant snapshot (orijinal backup mantığı uyarlanır) |

## 3. Mimari (hedef)

```
┌───────────────────────────── Ev makinesi / Coolify ─────────────────────────────┐
│                                                                                 │
│  ┌───────────────┐   REST + SSE    ┌──────────────┐   iş kuyruğu    ┌─────────┐ │
│  │ web (SPA)     │ ◄─────────────► │ api (Flask)  │ ──────────────► │ pipeline│ │
│  │ Vite+React    │                 │ auth, upload │   (dosya tabanlı│ worker  │ │
│  │ shadcn/ui     │                 │ status, chat │    basit kuyruk)│ parse→  │ │
│  └───────────────┘                 └──────┬───────┘                 │ chunk→  │ │
│                                           │                         │ vector  │ │
│                                    ┌──────▼──────┐                  └────┬────┘ │
│                                    │   qdrant    │◄─────────────────────┘      │
│                                    └─────────────┘                             │
│  /corpus (birim): BELGELER/ notlar/ parsed/ chunks/ vectorize/ yedekler/        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

İlkeler:

1. **Event-driven registry**: `document_nodes.json` artık klasör taramasıyla
   değil, **UI olaylarıyla** (upload/delete/edit-note) yazılır. Mevcut
   `classify_documents.py` (scan) bir defalık uzlaştırma (reconciliation)
   aracı olarak kalır; nightly zincirinde ZORUNLU değildir.
2. **Dosya yaşam döngüsü = tek kaynak**: upload → (parse → chunk → vectorize);
   değişiklik → eski türevlerin silinmesi + yeniden işleme; silme → tüm
   türevlerin silinmesi. Bu üçü tek "belge işLEYİCİ" soyutlamasında toplanır.
3. **Ön yüz ve API ayrımı**: Flask yalnız JSON + SSE servis eder; şablon arayüzü
   (webapp.py'deki HTML tarafı) kademeli olarak emekliye ayrılır.
4. **Maliyet farkındalığı**: 500+ belge ölçeğinde her upload ayrı işlenir;
   toplu yeniden işlemeye giren hiçbir kod yolu "tüm korpus"u varsayılan
   yapmaz.

## 4. Task Grupları

### Grup A — Belge yaşam döngüsü ve pipeline tetikleme (çekirdek)

| ID | Task | Kabul kriteri |
|----|------|---------------|
| A1 | **Upload API**: `POST /api/documents` (çoklu dosya, tip/boyut kontrolü, isim çakışması çözümü). Dosya `/corpus/BELGELER/`e yazılır, registry'e `NEW` kaydı düşer, iş kuyruğuna girer. | Çoklu yükleme; geçersiz tip reddi; aynı isimde `-1` türetme; registry tutarlı |
| A2 | **Tek-dosya pipeline koşucu**: mevcut parse→chunk→vectorize aşamalarının tek `doc_id` üzerinde çalıştırılması (mevcut `--doc`/incremental kapıları kullanılır). VLM yoğun taranmış PDF'lerde sayfa bazlı ilerleme state'e yazılır. | Tek dosya upload'ı diğerlerini işlemez; ilerleme % sayfa bazlı okunabilir |
| A3 | **Silme**: `DELETE /api/documents/{id}` → `forget_deleted_source` mantığı uyarlanır: parsed klasörü, chunk'lar, Qdrant noktaları, registry kaydı temizlenir. | Silinen dosyanın hiçbir izi (chunk/vektör/kütüphane satırı) kalmaz |
| A4 | **Değişiklik**: aynı rel_path ile yeniden upload (content_hash farklı) → eski türevler silinir (A3 yolu), yeni işleme (A2) tetiklenir. "Eski hali silinmiş + yeni hali eklenmiş" semantiği. | Değişen dosyada eski chunk'lara chat cevabı veremez; yeni içerik bulunur |
| A5 | **Durum deposu + API**: dosya başına `queued/parsing(%)/chunking/vectorizing/ready/error(+sebep)` durumu; `GET /api/documents` ve `GET /api/documents/{id}/status` + SSE akışı. | Rozet verisi API'den okunabilir; hata durumunda insan-okur sebep |
| A6 | **İş kuyruğu**: redis'siz basit mekanizma — api içi **sınırlı iş parçacığı havuzu** (öneri: max 2 eşzamanlı, tek VLM'yi dolaşan sıra). Uzun taranmış PDF'lerde iptal desteği. | İki eşzamanlı upload sıralanır; api restart'ta "queued" işler yeniden keşfedilir (kesinti toparlama) |
| A7 | **Notlar → korpus**: `notlar/` dizini txt notları için aynı yaşam döngüsüne bağlanır (doc_type=NOTE). Not kaydedilince eski chunk'ları silinip yenileri vektöre yazılır. | Not düzenleme sonrası chat o notun yeni halini kaynak gösterir |

### Grup B — Ön yüz (Vite + React + shadcn/ui)

| ID | Task | Kabul kriteri |
|----|------|---------------|
| B1 | **Scaffold**: Vite + React + TS + Tailwind + shadcn/ui; klasör yapısı `web/`; Flask'ın `/` altında statik servis etmesi (tek origin, CORS yok) | `docker compose up` ile SPA açılır |
| B2 | **Tema sistemi**: Wada Sanzo paleti CSS değişkenleri (E grubu), açık/koyu toggle (localStorage + `prefers-color-scheme`) | İki temada da WCAG AA kontrast |
| B3 | **Login ekranı** (K3): tek şifre, httpOnly oturum çerezi, uçtan uca auth middleware | Şifresiz her API 401 döner |
| B4 | **Kütüphane görünümü**: belge listesi (ad, tip, tarih, boyut, durum rozeti), sürükle-bırak çoklu yükleme, silme (onaylı), hata detayı gösterimi, arama/filtre | K16 rozetleri canlı güncellenir (SSE) |
| B5 | **Chat görünümü**: mesaj akışı, streaming cevap (SSE), sohbet geçmişi kalıcılığı, yeni sohbet | Sayfa yenilenince geçmiş durur; akış kanıta kadar canlı |
| B6 | **Evidence bileşenleri** (K14/K15): satır içi `[n]` rozetleri, cevap sonu numaralı kaynak listesi (belge + sayfa + bölüm), rozete tıklayınca ilgili kaynağa kaydırma, çelişki uyarı kutusu, "hekime danışın" dipnotu | Tıbbi cevapsız atıf üretilemez; çelişki örneğinde uyarı görünür |
| B7 | **Belge görüntüleyici** (K13): `react-markdown` + typography/pretty-render (tablo, kod, liste, başlık hiyerarşisi), sayfa referansı çapaları, "orijinali gör" (pdf.js / tarayıcı gömülü viewer) | Taranan kitap bölümü okunabilir biçimde render olur; evidence tıklaması ilgili bölüme götürür |
| B8 | **Notlar modülü**: not listesi, oluştur/düzenle/sil (basit metin editörü), kaydet → A7 tetiklenir, "chat'e dahil" durumu görünür | Not düzenlemesi ~çevrimiçi görünür süreçte vektöre işlenir |
| B9 | **API istemcisi + tipler**: tip güvenli fetch katmanı, SSE yardımcıları, hata/toast standartları | Tek yerden API sürümleme |

### Grup C — Evidence & prompt katmanı (tıbbileştirme)

| ID | Task | Kabul kriteri |
|----|------|---------------|
| C1 | **doc_question stratejisi yeniden yazımı**: atıf isteğe bağlılıktan çıkarılır — her olgusal iddia için zorunlu `doc_id + sayfa + bölüm` atfı | Strategi prompt'u ve ilgili şablonlar güncellenir; örnek cevaplar kurala uyar |
| C2 | **Çelişki politikası**: aynı olgu için farklı değer veren chunk'larda ikisini de göster + "kaynaklar farklı bilgi veriyor" uyarısı; model tek değer uyduramaz | Sentetik çelişki korpunda test edilir |
| C3 | **Bulunamadı + disclaimer**: kaynak yoksa uydurma yasak ("bulamadım" + öneri); tıbbi nitelikli her cevabın sonuna kısa hekim-danışma notu (K15) | Boş retrieval'da disclaimer'sız sahte cevap üretilemez |
| C4 | **Dil politikası** (K4): soru diliyle cevap; TR/EN karışık chunk'larda terim sadakati (İngilizce tıbbi terim + Türkçe açıklama) | TR soruya TR cevap; EN kaynak alıntısı korunur |
| C5 | **Router sadeleştirme**: neredeyse tüm intent'ler `default_topn`'e; kullanılmayan strateji prompt'ları arşivlenir (silinmez) | Router tablosu config'te gözden geçirilmiş |

### Grup D — Tasarım sistemi (Wada Sanzo: fildişi + güneş)

| ID | Task | Kabul kriteri |
|----|------|---------------|
| D1 | **Palet token'ları**: Wada Sanzo'dan fildişi zemin ailesi (örn. Gofun/Torinoko krem hattı) + güneş vurguları (Yamabuki sarısı, Kaki turuncusu, koyu kızıl); shadcn CSS değişkenleri (primary/secondary/accent/destructive/muted/ring) | Palet tablosu PLAN'da hex değerleriyle listelenir; shadcn temasına uygulanır |
| D2 | **Koyu tema eşlemesi**: aynı hue'ların koyu zemin karşılıkları (Sumi mürekkep zemin + sıcak vurgular) | Koyu temada vurgular aynı kimliği taşır |
| D3 | **Bileşen cilası**: durum rozetleri, evidence kutuları, toast, boş-durum (empty state) ekranları paletle uyumlu | Tüm sayfalar tek tasarım dilinde |

> D1 palet taslağı (hex'ler uygulama sırasında kalibre edilir):
> `--background` fildişi `#FAF6EE` · `--foreground` mürekkep `#2B2926` ·
> `--primary` Yamabuki `#E8A020` · `--accent` Kaki `#D96C2C` ·
> `--secondary` çay yeşili muadili nötr `#8C8473` · `--destructive` kızıl `#B3352C`.
> Koyu: zemin `#201D1A`, kart `#2A2622`, aynı vurgular açık tonlarda.

### Grup E — Deploy & operasyon

| ID | Task | Kabul kriteri |
|----|------|---------------|
| E1 | **Compose güncellemesi**: `web` (SPA statik + nginx veya Flask statik), `api`, `pipeline-worker` (A6 iş parçacıkları api'de ise api ile birleşir — karar noktası), `qdrant`. Coolify uyumlu env/volume tanımları | Sıfırdan `docker compose up -d` çalışır |
| E2 | **Volume şeması**: `/corpus` tek yazıcı düzeni upload API'sine göre güncellenir (api artık YAZICI — `:ro` kuralı gözden geçirilir); ayrıcalıklar uid 1000 ile uyumlu | Upload sonrası pipeline ve api aynı dosyayı görür |
| E3 | **Gecelik yedek** (K17): korpus + notlar + Qdrant snapshot → `/corpus/yedekler/`; mevcut `nightly_backup` mantığı uyarlanır; geri yükleme prosedürü yazılır | Yedekten geri yükleme tatbikatı başarılı |
| E4 | **Gözlemlenebilirlik**: upload/işleme olayları log; panel (lineage) med-rag akışına uyarlanır ya da kapsamdan çıkar (karar noktası) | Hatalı belge logdan teşhis edilebilir |
| E5 | **Güvenlik sertleştirme**: tek şifre + oturum; upload boyut limiti; uzantı whitelist; ev ağı dışına açmama notları (Coolify/reverse proxy) | Şifresiz erişim ve büyük dosya DoS kapalı |

### Grup F — Kalite & kabul

| ID | Task | Kabul kriteri |
|----|------|---------------|
| F1 | **Mevcut offline suite korunur**: baseline 61 önceden-var-olan fail dışında yeni fail YOK; import-linter sözleşmeleri güncel | `pytest src/ tools/ -q` fork baseline'ında kalır |
| F2 | **Uçtan uca kabul senaryosu**: yükle → durum izle → hazır → soru sor → atıflı cevap → kaynağa tıkla → belge görüntüleyici açılır → not ekle → not kaynaklı cevap → dosyayı sil → derhal cevabı kaybolur | Senaryonun tamamı el ile koşulur ve kayda geçer |
| F3 | **Değişiklik/silme testleri**: A3/A4'ün synthetic corpus üzerinde otomatik testleri | Otomatik testler yeşil |
| F4 | **Parser kalite ölçümü**: `tools/benchmark` ile taranmış PDF örneklemi puanlanır (tıbbi tablolar/doz satırları odaklı scope dosyası) | Doz tablosu örneklerinde tablo bütünlüğü doğrulanır |
| F5 | **Dokümantasyon**: README + CONFIG.md yeni mimariye göre güncellenir; kullanıcıya 1 sayfalık kullanım kılavuzu | Yeni kullanıcı kurulumdan kullanıma self-servis |

## 5. Uygulama sırası (önerilen)

1. **A1→A6** (belge yaşam döngüsü + durum) — çekirdek değeri taşır
2. **C1→C5** (evidence katmanı) — chatbot zaten mevcut olduğundan erken kazanım
3. **B1+B2+D1→D3** (SPA iskeleti + tema) → **B4** (kütüphane) → **B5+B6** (chat+evidence)
4. **B7** (görüntüleyici) → **B8** (notlar, A7 ile birlikte)
5. **E1→E5** (deploy/backup sertleştirme) → **F2** (uçtan uca kabul)

## 6. Açık Karar Noktaları (uygulama sırasında netleşir)

| # | Konu | KARAR |
|---|------|-------|
| Açık-1 | İş kuyruğu konumu | **Ayrı `pipeline-worker` servisi**: dosya-tabanlı kuyruk (`/corpus/isler`), tek replika, sıralı koşum; api yalnız iş DOSYASI yazar (import yok). Kesinti toparlama worker restart'ıyla. |
| Açık-2 | Scan/nightly rolü | Yaşam döngüsü **event-driven** (UI olayları yazar); `classify_documents.py` uzlaştırma aracı olarak durur, zincirde ZORUNLU değil. |
| Açık-3 | Embedding modeli | Mevcut config'teki model **korunur**. Değişiklik istenirse: koleksiyon silinip yeniden kurulmalı (üç taraf hizalı) — yordam `DEPLOY.md` §1'de. |
| Açık-4 | Panel (lineage) kaderi | **Kapsam dışı**: kod duruyor (facts gibi dormant), compose'a bağlı değil; teşhis log + `durum/` + gecelik raporlarla yürür (`DEPLOY.md` §4). |
| Açık-5 | Upload üst limitleri | Dosya başına **200 MB** (`MEDRAG_MAX_YUKLEME_MB` ile aşılır), uzantı whitelist'i; istek başına dosya sayısı sınırsız (tek kullanıcı). |
| Açık-6 | Chat geçmişi saklama | `conversation_log` **kalıcı** (`/logs/conversations` volume'u); okuma ucu `/api/chat/history`, oturum çereziyle; kullanıcı "Yeni sohbet"le ayrılır, eski kayıt diskte kalır. |

## 7. İzlenebilirlik: beklenti → task

| Kullanıcı beklentisi | Task |
|----------------------|------|
| "dosya ekledikçe corpus büyüyecek, pipeline tetiklenir" | A1, A2, A6, B4 |
| "dosya çıkartmada türevler silinir" | A3 |
| "değişen dosya = eski sil + yeni ekle" | A4 |
| "wada sanzo renkleri, shadcn, renkli kombinasyon" | D1–D3, B2 |
| "kütüphane + chat + notlar + belge gezinme" | B4, B5, B7, B8, A7 |
| "kullanıcı-dostu markdown render" | B7 |
| "evidence her cevapta kritik" | C1–C3, B6 |
| "basit şifre" | B3, E5 |
| "coolify ile ev makinesinde" | E1, E2 |
| "gecelik yedek" | E3 |
