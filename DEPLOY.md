# med-rag — Deploy & Operasyon Kılavuzu

> Hedef: ev makinesinde (Coolify ya da düz docker compose) **bu gece**
> çalışır bir kurulum. Kurulum → yedek → geri yükleme → güvenlik, hepsi bu
> dosyada. Ortam değişkenlerinin tam kataloğu [`CONFIG.md`](CONFIG.md)'de,
> kullanım kılavuzu [`KULLANIM.md`](KULLANIM.md)'dedir.

## 1. Hızlı kurulum (docker compose)

```bash
cp .env.example .env          # doldur: MEDRAG_SIFRE + LLM/embedding değerleri
docker compose up -d --build  # web + pipeline-worker + pipeline + qdrant
```

Dört servis:

| Servis | Görev |
|---|---|
| `web` | SPA + API (gunicorn, port 8507). Upload, chat, kütüphane, notlar. |
| `pipeline-worker` | İş kuyruğu (`/corpus/isler`): upload edilen dosyayı parse→chunk→vectorize ile işler, silme/temizlik yapar. Tek replika, sıralı koşum. |
| `pipeline` | Boşta bekleyen (`sleep infinity`) `docker exec` hedefi — gecelik zincir ve yedek bu konteynıra exec ile koşturulur. |
| `qdrant` | Vektör deposu (sabit sürüm `v1.19.0`). |

Sağlık kontrolü:

```bash
docker compose ps                 # web healthy olmalı
curl -s http://localhost:8507/api/auth/session
```

Tarayıcı: `http://<makine-ip>:8507` → şifre ekranı → kütüphane.

### İlk çalıştırma

1. `.env`'de `MEDRAG_SIFRE`'yi mutlaka doldurun (boşsa auth **kapalı** olur).
2. LLM/embedding adresleri host'ta çalışıyorsa `http://host.docker.internal:...`
   kullanın (Linux'ta `extra_hosts` compose'da zaten tanımlı).
3. Kütüphaneden ilk belgeyi yükleyin; durum rozeti `kuyrukta → işleniyor →
   hazır` akışını izler. İlk işleme embedding modelini indirir (yavaş olabilir).
4. **Kritik hizalama:** `EMBEDDING_MODEL` + Qdrant koleksiyon adı üç tarafta
   (vectorize, retrieval, chatbot) aynı olmalıdır. Model değiştirileceği zaman:
   koleksiyonu silip yeniden oluşturun (`curl -X DELETE
   http://localhost:6333/collections/<ad>`), sonra tüm belgeleri yeniden yükleyin.

## 2. Yerleşim (volume şeması)

Tek kalıcı kök: `CORPUS_HOST_DIR` (ör. `/home/user/med-rag/korpus`):

```
korpus/
  BELGELER/            kaynak belgeler + notlar/ (TEK GERÇEK KAYNAK)
  document_nodes.json  registry (event-driven; upload/silme yazar)
  parsed/              IR + markdown türevleri
  chunks/              all_chunks.json
  vectorize/           embedding durum/state
  durum/               dosya başına işleme durumu (rozeti besler)
  isler/               iş kuyruğu (dosya tabanlı)
  storage/images       parser blob deposu (SILEN DATA LOSS noktası — kalıcı olmalı)
  yedekler/            gecelik yedekler (aşağıda)
loglar/
  chatbot/             uygulama logları
  conversations/       sohbet kayıtları (hesap verebilirlik izi)
```

`web` ve `pipeline-worker` korpusa **okuma-yazma** erişir (api bir yazardır:
upload → BELGELER + registry + isler/). Gecelik zincir yalnız `pipeline`
konteynerinden koşturulur.

## 3. Gecelik yedek (K17/E3)

Yedek iki katman alır:

1. **Altyapı** (`nightly_backup.run_backup`): registry, chunks, parse çıktısı,
   specs.db + **Qdrant snapshot** (sunucu taraflı; `qdrant/snapshot_name.txt`).
2. **med-rag eki**: kaynak korpus `BELGELER/` (belgeler + notlar) ve `durum/`.

Koşturma (cron ya da Coolify scheduled task — hedef: `pipeline` konteyneri):

```bash
docker exec med-rag-pipeline-1 python -m medrag.pipeline.lifecycle.backup_cli
```

Yedek kökü: `NIGHTLY_BACKUP_ROOT` (compose'da `/corpus/yedekler` yerine
`/corpus/nightly_backups`'a bağlanmıştır — host'ta
`$CORPUS_HOST_DIR/nightly_backups`). Cron örneği (her gece 03:00):

```cron
0 3 * * * docker exec med-rag-pipeline-1 python -m medrag.pipeline.lifecycle.backup_cli >> /home/user/med-rag/loglar/backup.log 2>&1
```

### 3.1. Geri yükleme prosedürü (tatbikat adımları)

> İlke: restore, yedek alındığı andaki hâle döner. Türevler (parsed/chunks)
> bozuk yedekten geri gelse bile kaynak `BELGELER/` hep elinizedir.

```bash
# 0) Dur: worker'ı durdur (yeni iş almasın), web kalsın da sonrasın da test edebilelim
docker compose stop pipeline-worker

# 1) En son yedeği belirle
YEDEK=$(ls -1d $CORPUS_HOST_DIR/nightly_backups/*/ | sort | tail -1); echo $YEDEK
cat "$YEDEK/manifest.json"   # eksik/kayıp var mı bak

# 2) Altyapı türevlerini geri yaz (registry, chunks, parsed, specs.db)
docker exec med-rag-pipeline-1 python -c "
from medrag.pipeline.cli import nightly_backup
nightly_backup.restore_backup('$YEDEK')
"
# Not: qdrant kalemi burada ATLANIR (snapshot sunucu tarafındadır) — adım 3.

# 3) Qdrant snapshot'ını geri yükle
SNAP=$(cat "$YEDEK/qdrant/snapshot_name.txt")
# 3a) Snapshot dosyası qdrant volume'ünde değilse önce kopyala:
#     docker cp "$YEDEK/qdrant/$SNAP" med-rag-qdrant-1:/qdrant/storage/snapshots/
docker exec med-rag-qdrant-1 curl -s -X PUT \
  http://localhost:6333/collections/medrag_chunks/snapshots/recover \
  -H 'Content-Type: application/json' \
  -d "{\"location\": \"snapshots/$SNAP\"}"

# 4) Kaynak korpus + notlar + durum (med-rag eki) — elle geri kopyala
rsync -a --delete "$YEDEK/BELGELER/" "$CORPUS_HOST_DIR/BELGELER/"
rsync -a "$YEDEK/durum/"      "$CORPUS_HOST_DIR/durum/"

# 5) Worker'ı geri başlat; kuyruk temiz olsun
rm -f "$CORPUS_HOST_DIR"/isler/*.json "$CORPUS_HOST_DIR"/isler/*.claim 2>/dev/null
docker compose start pipeline-worker

# 6) Tatbikat doğrulaması
curl -s http://localhost:8507/api/library/documents | head -c 400
# → belge listesi geliyorsa restore tamam. Bir belge açıp chat'te
#   soru sor; atıf geliyorsa Qdrant restore de sağlam demektir.
```

`restore_backup`, sqlite'ı WAL-güvenli kopyayla (`_restore_sqlite`) geri
yazar; qdrant hariç tüm kalemleri manifest sırasıyla döndürür.

## 4. Gözlemlenebilirlik (E4)

| Ne | Nerede |
|---|---|
| Uygulama logları (api) | `$LOGS_HOST_DIR/chatbot/` |
| Sohbet kayıtları | `$LOGS_HOST_DIR/conversations/` (tek hesap verebilirlik izi) |
| İşleme durumu | `$CORPUS_HOST_DIR/durum/<doc_id>.json` (+ UI rozetleri, SSE) |
| Hatalı belge teşhisi | worker logları: `docker logs med-rag-pipeline-worker-1` |
| Gecelik raporlar | `$CORPUS_HOST_DIR/reports/nightly_YYYY-MM-DD.json` |

Eski lineage paneli (`src/medrag/api/panel/`) **kapsam dışı** bırakıldı: kod
duruyor ama compose'a servis olarak bağlı değil (karar: Açık-4). Teşhis yukarıdaki
log + durum dosyaları üzerinden yürür.

## 5. Güvenlik sertleştirme (E5)

- **Şifre**: `MEDRAG_SIFRE` boşsa auth kapalı — asla boş bırakmayın. Oturum,
  imzalı httpOnly çerezdir (`MEDRAG_GIZLI_ANAHTAR` ile; boşsa şifreden türetilir).
- **Upload sınırları**: uzantı whitelist'i (`.pdf .docx .pptx .xlsx .html .md`)
  dışındaki dosyalar 415 alır; dosya başına limit `MEDRAG_MAX_YUKLEME_MB`
  (varsayılan 200 MB), aşanlar 413 alır.
- **Ağ**: bu kurulum **ev ağı içindir**. Router'dan port yönlendirme YAPMAYIN;
  dışarıdan erişim gerekiyorsa Coolify/Traefik arkasında HTTPS + ek kimlik
  katmanıyla açın. `web` 8507'de tüm arayüzleri dinler — sadece LAN'ın
  görebildiğinden emin olun (gerekirse compose'da `ports: "127.0.0.1:8507:8507"`).
- **Sırlar**: yalnız `.env`'de (gitignore'ludur, commit edilmez).

## 6. Uçtan uca kabul (F2)

Otomatik duman testi (çalışan stack'e karşı):

```bash
.venv/bin/python tools/e2e_smoke.py --base-url http://localhost:8507 \
    --sample tests_ornek.pdf            # ör. küçük bir PDF
# şifre varsa: --password ...
# LLM dahil tam tur (yavaş, token harcar): --chat
```

Senaryo: yükle → durum izle → hazır → içerik gör → not ekle → not kaynaklı
arama → belge sil → anında kaybolduğunu doğrula. `--chat` ile: soru sor →
atıflı cevap → kaynağa tıkla (manuel adım).

El ile kabul çeki listesi (kayıt için):

1. [ ] Şifresiz istek `/api/*` uçlarında 401 alıyor
2. [ ] Çoklu dosya sürükle-bırak yükleme çalışıyor
3. [ ] Durum rozetleri canlı ilerliyor (kuyrukta → işleniyor → hazır)
4. [ ] Soruya cevap satır içi `[n]` rozetleriyle geliyor; rozet kaynağa götürüyor
5. [ ] Kaynak bulunamayan soruda "bulamadım" + hekim notu görünüyor
6. [ ] Not oluşturma → ~çevrimiçi işlenme → chat notu kaynak gösteriyor
7. [ ] Belge silme → kütüphane satırı, chunk, vektör izleri anında gidiyor
8. [ ] Aynı dosyanın değiştirilmiş hâlini yükleme → eski içerik artık bulunmuyor
9. [ ] Yedek koşuyor ve §3.1 tatbikatı başarıyla tamamlandı
