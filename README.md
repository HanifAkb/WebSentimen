# Prediksi Sentimen (Django + KNN/SVM)

Aplikasi web Django untuk analisis sentimen biner (`Positive`/`Negative`) memakai model pre-trained:

- `knn_model.joblib`
- `svm_linear_model.joblib` (fallback didukung: `svm_rbf_model.joblib`)

Website:

- https://analisissentimen.up.railway.app/

## Fitur Utama

- Login wajib untuk akses fitur utama
- Admin-only pembuatan akun baru
- Prediksi kalimat tunggal
- Prediksi batch dari file CSV/TXT
- Scraping Web X via `twitterapi.io` (API key dari user)
- Hasil KNN dan SVM berdampingan
- Riwayat scraping dan prediksi per user

## Cara Pakai

### 1) Buat Prediksi (`https://analisissentimen.up.railway.app/predict/`)

- Input 1 kalimat, atau upload 1 file (`.csv` / `.txt`)
- CSV:
  - auto-detect kolom teks: `text`, `tweet`, `content`, `sentence`
  - atau isi manual kolom teks
- TXT: setiap baris non-kosong dianggap 1 data
- Output:
  - label KNN dan SVM
  - skor jika tersedia
  - preview batch + unduh CSV hasil klasifikasi

### 2) Scraping Web X (`https://analisissentimen.up.railway.app/scraping/`)

- Isi API key, kueri, bahasa (opsional), tanggal mulai-selesai
- Hasil scraping diklasifikasikan oleh KNN + SVM
- Auto-continue tersedia saat proses belum selesai
- Dashboard otomatis tampil saat status scraping sudah `Selesai`
- API key disimpan sementara di browser (`sessionStorage`), tidak disimpan ke database

### 3) Riwayat (`https://analisissentimen.up.railway.app/history/`)

- Riwayat scraping dan prediksi dipisah per user
- Tabel riwayat sudah dipaginasi (10 data per halaman)

## Catatan Model
- Batas upload:
  - `SENTIMENT_UPLOAD_MAX_SIZE` (default 10 MB)
- Skor:
  - KNN: `predict_proba` (rentang `0-1`)
  - SVM: `decision_function` yang di-clip ke rentang `-1..1`
- Aturan label `Neutral`:
  - KNN: jika skor `0.45` s/d `0.55`
  - SVM: jika skor `-0.10` s/d `0.10`

## Keamanan

- Validasi upload extension/type/size
- CSRF aktif di form
- Validasi path aman untuk download CSV
- API key tidak dipersist ke DB
- Password user disimpan dalam bentuk hash Django (bukan plaintext)

## Instalasi Lokal

1. Clone repo lalu masuk ke folder project.
2. Buat virtual environment Django:

```bash
python -m venv .venv
```

3. Aktifkan virtual environment.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Bash:

```bash
source .venv/bin/activate
```

4. Install dependency:

```bash
pip install -r requirements.txt
```

5. Salin konfigurasi environment:

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

6. (Opsional) Letakkan/ganti file model di:

```text
sentiment_site/models/knn_model.joblib
sentiment_site/models/svm_linear_model.joblib
```

Model default sudah tersedia di repository. Langkah ini diperlukan jika Anda ingin mengganti model bawaan dengan model Anda sendiri.

Alternatif SVM yang juga didukung:

```text
sentiment_site/models/svm_rbf_model.joblib
```

Jika model bukan pipeline end-to-end, tambahkan artifact:

```text
sentiment_site/models/vectorizer.joblib
atau
sentiment_site/models/tfidf_vectorizer.joblib
sentiment_site/models/label_encoder.joblib   # opsional
```

7. Jalankan migrasi database:

```bash
python manage.py migrate
```

8. (Opsional) Buat akun admin:

```bash
python manage.py createsuperuser
```

9. Jalankan server Django:

```bash
python manage.py runserver
```

Akses aplikasi pada URL yang tampil di terminal setelah `runserver` berjalan (default biasanya `http://127.0.0.1:8000/`).

## Testing

Jalankan:

```bash
python manage.py test
```

Cek konfigurasi cepat:

```bash
python manage.py check
```

## Troubleshooting Singkat

- Error `No module named 'imblearn'`: install ulang dependency dari `requirements.txt`
- Error model butuh vectorizer: tambahkan `vectorizer.joblib` atau `tfidf_vectorizer.joblib`
- Scraping terasa lambat: perkecil rentang tanggal atau turunkan batas tweet per scraping via env
