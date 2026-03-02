# Django Sentiment Site (KNN + SVM)

Minimal Django web app for binary sentiment analysis (`Positive` / `Negative`) with two pre-trained models:

- `knn_model.joblib`
- `svm_rbf_model.joblib`

The app supports:

- login system (company mode)
- admin-only user registration
- single sentence prediction
- batch prediction from CSV/TXT upload
- tweet fetch from `twitterapi.io` using user-provided API key
- side-by-side KNN/SVM outputs
- persistent history per logged user (scraping + prediction)

## 1) Project Structure

```text
.
|-- manage.py
|-- requirements.txt
|-- README.md
|-- sentiment_site/
|   |-- settings.py
|   |-- urls.py
|   |-- models/
|       |-- knn_model.joblib            # place here
|       |-- svm_rbf_model.joblib        # place here
|       |-- vectorizer.joblib (optional)
|       |-- tfidf_vectorizer.joblib (optional)
|       |-- label_encoder.joblib (optional)
`-- sentiment_app/
    |-- forms.py
    |-- urls.py
    |-- views.py
    |-- services/
    |   |-- preprocess.py
    |   |-- model_service.py
    |   |-- file_service.py
    |   `-- twitter_client.py
    |-- templates/sentiment_app/
    |   |-- base.html
    |   |-- login.html
    |   |-- register.html
    |   |-- history.html
    |   |-- history_predict_detail.html
    |   |-- predict.html
    |   |-- beranda.html
    |   `-- twitter.html
    |-- static/sentiment_app/css/styles.css
    `-- tests/
        |-- test_preprocess.py
        |-- test_file_service.py
        |-- test_model_service.py
        `-- test_auth_history.py
```

## 2) Setup

1. Create and activate virtual environment.
2. Copy env template and adjust values:

```bash
cp .env.example .env
```

Untuk Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place model files into:

```text
sentiment_site/models/knn_model.joblib
sentiment_site/models/svm_rbf_model.joblib
```

The loader also checks project root as fallback (for quick local testing), but `sentiment_site/models/` is the recommended location.

5. Optional artifacts if your model is classifier-only (not pipeline):

```text
sentiment_site/models/vectorizer.joblib
or
sentiment_site/models/tfidf_vectorizer.joblib

sentiment_site/models/label_encoder.joblib   # optional
```

6. Run migrations:

```bash
python manage.py migrate
```

7. Create superuser (required to create other company accounts):

```bash
python manage.py createsuperuser
```

8. Start server:

```bash
python manage.py runserver
```

9. Open:

```text
http://127.0.0.1:8000/
```

## 3) How to Use

### Login (`/login/`)

- Semua halaman utama diproteksi login.
- Hanya admin/superuser yang bisa membuka halaman register akun baru (`/register/`).

### Home / Predict (`/predict/`)

- Type one sentence and submit, or upload one file (`.csv` / `.txt`).
- CSV:
  - auto-detect text column names: `text`, `tweet`, `content`, `sentence`
  - or fill "CSV text column" manually.
- TXT:
  - each non-empty line is one sample.
- Output:
  - KNN and SVM labels shown side-by-side.
  - Score shown when available:
    - `predict_proba` positive class probability when supported.
    - if unavailable and model has `decision_function`, score is sigmoid-scaled confidence-like value (not calibrated probability).
  - For batch prediction: preview first 20 rows + full CSV download.

### Scraping Web X (`/scraping/`)

- Input:
  - API key (not stored in DB; used in-memory for request only)
  - query
  - optional language
- Output:
  - classified preview (KNN/SVM) + pagination
  - hasil scraping bersifat sementara (tidak disimpan ke `media/outputs`)
  - refresh pada halaman hasil akan membersihkan output
  - setiap scraping yang berhasil otomatis disimpan ke riwayat user yang login

### Riwayat (`/history/`)

- Menampilkan riwayat scraping dan riwayat prediksi milik user login.
- Riwayat prediksi mencakup:
  - kalimat tunggal
  - unggah CSV/TXT
- Data history terisolasi per-user.
- Klik `Lihat` untuk membuka detail setiap riwayat.

Contoh kueri lanjutan (sesuaikan dukungan endpoint twitterapi.io Anda):

- `"frasa persis"` untuk exact phrase
- `(A OR B)` untuk alternatif keyword
- `-kata` atau `-filter:retweets` untuk exclude
- `from:username` untuk akun tertentu
- `min_faves:10`, `min_retweets:5` untuk minimum engagement
- `has:links`, `has:media` untuk tipe konten

Catatan: rentang tanggal sudah diatur dari field tanggal pada form, jadi tidak perlu menambahkan `since:`/`until:` di kueri.

## 4) Security Notes

- Upload validation:
  - extensions restricted to `.csv` / `.txt`
  - content type checked
  - max upload size default `10 MB` (configurable via env `SENTIMENT_UPLOAD_MAX_SIZE`)
- Download route:
  - filename regex validation
  - strict path resolution under `MEDIA_ROOT/outputs` (path traversal blocked)
- CSRF protection enabled in all forms
- Django messages framework used for safe UI feedback
- API key is never persisted
- Semua route inti diproteksi autentikasi (`login_required`)
- Registrasi akun baru dibatasi hanya superuser/admin
- Test keamanan auth mencakup skenario payload login mirip SQL injection
- Penyimpanan kredensial:
  - `username` disimpan sebagai teks biasa di database (untuk identitas login)
  - `password` tidak disimpan plain text, tetapi di-hash satu arah (PBKDF2/Scrypt)
- Hardening production disediakan lewat env:
  - `DJANGO_DEBUG=0`
  - `DJANGO_SECRET_KEY` wajib diisi
  - `DJANGO_ALLOWED_HOSTS` wajib diisi
  - opsi cookie/SSL/HSTS ada di `.env.example`

## 5) Twitter API Endpoint Notes

Twitter API wrapper is centralized in:

- `sentiment_app/services/twitter_client.py`

Key constants to edit:

- `BASE_URL`
- `SEARCH_ENDPOINT`

If your `twitterapi.io` plan uses different endpoint/params, update these constants and request parameter mapping there.

## 6) Model Loading and Behavior

- Models load lazily and are cached in-memory.
- App tries direct `model.predict([text])` first.
- If direct inference fails (usually means vectorization required), app tries:
  - preprocessing (`lowercase`, URL/user/hashtag cleanup, whitespace normalize)
  - vectorizer transform from `vectorizer.joblib` or `tfidf_vectorizer.joblib`
- If vectorizer artifacts are missing, a friendly UI error is shown.

Label mapping rules:

- numeric outputs: `1 -> Positive`, `0 -> Negative`
- string outputs: normalized to `Positive`/`Negative` when recognizable
- optional `label_encoder.joblib` is used if available

## 7) Tests

Run:

```bash
python manage.py test
```

Included tests:

- preprocessing behavior
- CSV parsing + text column detection
- prediction flow using mocked model artifacts
- auth/access control + history persistence (scraping + prediction)
- login payload SQL-like injection should fail authentication

## 8) Manual Test Checklist

1. Open `/` and submit a single sentence. Confirm both KNN and SVM labels appear.
2. Upload valid CSV with `text` column. Confirm preview + CSV download works.
3. Upload valid TXT with multiple lines. Confirm batch predictions.
4. Upload file > 10 MB. Confirm friendly validation error.
5. Upload unsupported extension (`.xlsx`). Confirm rejection.
6. Use CSV without usable text column and no manual column. Confirm clear error.
7. Remove/rename model files and submit. Confirm missing-model error message.
8. Use classifier-only model without vectorizer artifact. Confirm vectorizer-required message.
9. Open `/scraping/`, use API key + query, fetch tweets, then verify preview + pagination appears.
10. Try invalid API key and confirm error handling.
11. Verify `/download/<filename>/` blocks invalid filename/path attempts.

## 9) Deploy (Railway Manual)

Catatan: Railway menggunakan model free trial/credit, lalu berbayar setelah limit tercapai.

### A. Persiapan repo

1. Pastikan file model ada di repo:
   - `sentiment_site/models/knn_model.joblib`
   - `sentiment_site/models/svm_rbf_model.joblib`
2. Pastikan push terbaru sudah masuk (termasuk `requirements.txt`, `Procfile`, `build.sh`).

### B. Buat project dan service

1. Buka Railway Dashboard -> `New Project`.
2. Pilih `Deploy from GitHub Repo`, lalu pilih repo ini.
3. Setelah service web terbentuk, tambah database:
   - `New` -> `Database` -> `Add PostgreSQL`.

### C. Konfigurasi web service (Settings)

Set command berikut di service web:

- Build Command:
  - `bash build.sh`
- Start Command:
  - `gunicorn sentiment_site.wsgi:application --bind 0.0.0.0:$PORT`
- Pre-Deploy Command:
  - `python manage.py migrate --noinput`

### D. Environment variables (Variables tab)

Set variabel berikut di service web:

- `DJANGO_SECRET_KEY` = string acak panjang
- `DJANGO_DEBUG` = `0`
- `DJANGO_SESSION_COOKIE_SECURE` = `1`
- `DJANGO_CSRF_COOKIE_SECURE` = `1`
- `DJANGO_SECURE_SSL_REDIRECT` = `1`
- `DJANGO_SECURE_HSTS_SECONDS` = `31536000`
- `DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS` = `1`
- `DJANGO_SECURE_HSTS_PRELOAD` = `1`

Untuk database:

1. Buat env `DATABASE_URL`.
2. Klik `Add Reference` -> pilih service PostgreSQL -> pilih `DATABASE_URL`.

`DJANGO_ALLOWED_HOSTS` dan `DJANGO_CSRF_TRUSTED_ORIGINS` bisa dikosongkan di Railway karena aplikasi otomatis membaca `RAILWAY_PUBLIC_DOMAIN`.

### E. Deploy pertama

1. Trigger deploy (otomatis saat push, atau klik `Deploy` manual).
2. Tunggu status `Success`.
3. Buka domain yang diberikan Railway.

### F. Buat akun admin pertama (superuser)

Gunakan Railway CLI dari lokal:

```bash
railway login
railway link
railway run python manage.py createsuperuser
```

Lalu login ke `/admin/` atau `/login/`.
