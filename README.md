# med-rag

> **Köken:** Bu depo, [`doc-rag-pipeline`](https://github.com/) adlı ürün-belge
> asistanından türetilmiştir (kopyala–sök–yeniden adlandır). Amaç farklıdır:
> **tıbbi içerikli PDF'lerden (dijital + taranmış/foto PDF) kanıt (evidence)
> göstererek cevap veren bir sohbet asistanı.** Hastalık bilgileri, ilaç dozları
> gibi konularda her cevabın hangi belge/sayfa/bölüme dayandığı kritiktir.

PDF, DOCX, PPTX, XLSX, HTML, Markdown formatındaki belgeleri ortak bir ara
temsile (IR) çevirir; görselleri OCR'dan geçirir; tabloları yapılandırır;
oluşan korpusu token-bazlı chunk'lara böler; vektör veritabanına (Qdrant)
yükler; ve bu kaynaktan soruları **kaynak atfıyla** cevaplayan bir chatbot
sunar. Model tarafı sağlayıcı-bağımsızdır (yerel Ollama / OpenAI-uyumlu /
OpenRouter / Anthropic).

## Orijinal projeden farklar

| Konu | doc-rag-pipeline | med-rag |
|---|---|---|
| Yapılandırılmış veritabanı (`specs.db` + text2sql) | Aktif yol | **Kod duruyor, kullanılmıyor** — ileride yeniden açılabilir |
| Excel ürün kataloğu (`chatbot-corpus/product_info/`) | Var | **Kaldırıldı** — scan "catalog-less" modda: katalog yoksa tüm belgeler `owner_ids=[]` ile kaydedilir |
| WhatsApp köprüsü + worker + Redis | Var | **Kaldırıldı** — tek sunum yüzeyi web UI |
| Nightly zinciri | products → scan → parse → chunk → ownership → facts → load → vectorize | scan → parse → chunk → ownership → facts → load → vectorize |
| Paket adı | `urun` | `medrag` |
| Kanıt politikası | Kaynak atfı çoğunlukla isteğe bağlı | Tıbbi kullanım için **zorunlu atıf** hedefi (strateji katmanında tıbbileştirme ilk iş) |

## Mimari (değişmedi)

```
 chatbot-corpus/                        src/medrag/pipeline/
 ──────────────                         ──────────────────
 BELGELER/ tree  ──► document_nodes.json (registry, catalog-less scan)
                                       │
                             ┌─────────▼─────────┐
                             │  cli: parse        │  parser: file → IR JSON + Markdown
                             │                    │  (dijital / hybrid / scanned yolları;
                             │                    │   görsel OCR, tablo yapısı + başlık)
                             └─────────┬─────────┘
                             ┌─────────▼─────────┐
                             │  cli: chunk        │  chunker: IR → token-bazlı,
                             │  all_chunks.json   │  yapıyı koruyan chunk'lar (LLM'siz)
                             └─────────┬─────────┘
                                       │
                             ┌─────────▼─────────┐
                             │  vectorize         │  chunk → embed → Qdrant upsert
                             └─────────┬─────────┘
                                       │  vektör yolu
                             ┌─────────▼─────────┐
                             │  src/medrag/api/   │  intent → deterministik router →
                             │  chatbot (web UI)  │  flow → cevap modeli (evidence paneli,
                             │                    │  SSE trace)
                             └───────────────────┘
```

**Katman kuralı:** hiçbir bileşen diğerini doğrudan import etmez; bağ dosya
sistemi ve veritabanıdır. `medrag.api` ↔ `medrag.pipeline` bağımsızlığı ve
`medrag.core`'un en altta olması `pyproject.toml` içindeki import-linter
sözleşmeleriyle zorlanır.

## Bileşenler

| Bileşen | Ne yapar | Kod |
|---|---|---|
| **chatbot-corpus** | `BELGELER/` ağacını tarayıp `document_nodes.json` registry üretir. Katalog'suz mod: ürün beyaz listesi yoksa her belge kaydedilir. | [`chatbot-corpus/document_info/`](chatbot-corpus/document_info/README.md) |
| **parser** | 6 format → ortak IR + Markdown. PDF için üç yol: deterministik (dijital), VLM-doğrulamalı (hybrid), render+VLM (**scanned/foto PDF**). Görsel OCR, tablo ızgarası PDF'in kendi vektör geometrisinden kurulur; model yalnız düşük güven bölgelerinde "hakem". | [`src/medrag/pipeline/parser/`](src/medrag/pipeline/parser/README.md) |
| **pipeline/cli** | Aşama orkestrasyonu + gecelik koşu (`medrag-nightly`). | [`src/medrag/pipeline/cli/`](src/medrag/pipeline/cli/README.md) |
| **chunker** | Token-bazlı, yapıyı koruyan chunk'lar. `doc`/`section`/`page` meta'sı taşır — kanıt atfının temeli. LLM'siz, tamamen offline. | [`src/medrag/pipeline/chunker/`](src/medrag/pipeline/chunker/README.md) |
| **vectorize** | Embedding → Qdrant upsert (delete-then-reinsert + bayatlık kapısı). | [`src/medrag/pipeline/vectorize/`](src/medrag/pipeline/vectorize/README.md) |
| **api (chatbot)** | Orkestrasyon: intent → router → flow (`default_topn` ana yol) → cevap modeli. Web UI + evidence paneli. | [`src/medrag/api/`](src/medrag/api/) |
| **api/retrieval** | Projeden bağımsız RAG retrieval katmanı: intent_classification, top_n (Qdrant), query_rewriting, raptor, db_query. | [`src/medrag/api/retrieval/`](src/medrag/api/retrieval/) · [`retrieval/ARCHITECTURE.md`](retrieval/ARCHITECTURE.md) |
| **facts (dormant)** | Evidence'lı spec çıkarımı → `specs.db` + text2sql yolu. med-rag'de **kullanılmaz** ama ileride yapılandırılmış yol istenirse kod hazırdır. | [`src/medrag/pipeline/facts/`](src/medrag/pipeline/facts/) |
| **panel** | Salt-okunur lineage/bayatlık paneli. | [`src/medrag/api/panel/`](src/medrag/api/panel/README.md) |
| **core** | Ortak altyapı: config yükleyici, sqlite yardımcıları, OpenAI-uyumlu LLM istemcisi. | [`src/medrag/core/`](src/medrag/core/) |
| **tools/benchmark** | Parser Markdown'ını kaynak PDF'e karşı VLM jüri ile puanlar. | [`tools/benchmark/`](tools/benchmark/README.md) |
| **packages/text2sql-native** | facts'ın kullandığı iki aşamalı text2sql motoru (dormant). | [`packages/text2sql-native/`](packages/text2sql-native/README.md) |

Kök dizinlerdeki `chunker/`, `vectorize/`, `facts/`, `chatbot/`, `retrieval/`,
`pipeline/` klasörleri **yalnızca veri/konfigürasyon/dokümantasyon** tutar;
kod `src/medrag/` altındadır.

## Kurulum

Python **≥ 3.11**.

```bash
uv sync --extra parse --extra serve --extra dev
# veya
pip install -e ".[parse,serve,dev]"
```

Model/embedding çağrıları HTTP üzerinden yapılır (yerel Ollama veya barındırılan
OpenAI-uyumlu uç nokta); bu makinede model çalıştırılmaz/indirilmez.

## Hızlı başlangıç

```bash
# Tek dosyayı parse et (scanned PDF dahil)
python src/medrag/pipeline/parser/scripts/to_markdown.py belge.pdf

# Aşama aşama
python -m medrag.pipeline.cli.run_parse_pipeline     # registry → IR + Markdown
python -m medrag.pipeline.cli.run_chunk_pipeline     # IR → all_chunks.json
python -m medrag.pipeline.vectorize                  # chunk → embed → Qdrant

# Gecelik zincir
medrag-nightly                # scan → parse → chunk → ownership → facts → load → vectorize
medrag-nightly --from chunk   # belirli aşamadan başla

# Chatbot
python -m medrag.api.webapp   # http://127.0.0.1:8507
```

Korpus düzeni: `BELGELER/` ağacı (PDF'ler) → `document_nodes.json` →
`parsed/` → `chunks/all_chunks.json` → Qdrant. Tüm yollar bileşen
`.env.example` şablonlarından ayarlanır (bkz. [`CONFIG.md`](CONFIG.md)).

**Kritik hizalama:** `EMBEDDING_MODEL` ve Qdrant koleksiyon adı vectorize,
retrieval ve chatbot tarafında bayt-bayt aynı olmalıdır.

## Test

Tüm test paketi **offline**'dır — ağ, model veya API anahtarı gerekmez:

```bash
pytest src/ tools/ -q
python -m pytest src/medrag/tests/test_import_contracts.py   # katman sınırı
```

## Yol haritası (tıbbileştirme)

1. `chatbot/strategies/doc_question.md` — atfı isteğe bağıl olmaktan çıkarıp
   **her iddia için zorunlu `doc_id + sayfa + section`** politikası.
2. `preamble.py` — "bulunamadıysa uydurma" davranışına tıbbi güvenlik notu
   eklenmesi (doz konularında hekime danışma uyarısı).
3. Router tablosunun sadeleştirilmesi (neredeyse her şey `default_topn`).
4. İstemci sözleşmelerinde ACME/ürün kalıntısı içeren prompt'ların gözden
   geçirilmesi.

## Deploy (Docker)

```bash
docker build --target pipeline -t medrag-pipeline:local .
docker compose up -d
```

Servisler: `web` (chat UI, 8507), `pipeline` (gecelik koşunun exec hedefi,
`sleep infinity`), `qdrant`. Korpus tek bir host dizininde
(`CORPUS_HOST_DIR` → `/corpus`): serve tarafı `:ro` bağlar, `pipeline` tek
yazıcıdır.
