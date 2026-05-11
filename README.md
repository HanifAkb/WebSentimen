# Sistem Analisis Sentimen (KNN + SVM)

Aplikasi web Django untuk analisis sentimen Bahasa Indonesia menggunakan model KNN dan SVM, dengan hasil gabungan `Soft Voting`.

Website:

- `https://analisissentimen.up.railway.app/`

## Ringkasan

Fitur utama aplikasi:

- login wajib untuk semua fitur utama
- prediksi `Kalimat`
- prediksi batch dari file `CSV/TXT`
- scraping Web X melalui `twitterapi.io`
- riwayat prediksi dan scraping per user
- unduh hasil klasifikasi dalam format CSV

## Halaman Utama

### 1. Beranda

Berisi ringkasan:

- total scraping
- total data hasil scraping
- total prediksi
- total data hasil prediksi
- panduan cara menggunakan menu utama

### 2. Buat Analisis

Mendukung dua mode:

- `Kalimat`
- `CSV/TXT`

### 3. Scraping Web X

User memasukkan:

- API key `twitterapi.io`
- query
- bahasa (opsional)
- tanggal mulai
- tanggal selesai

Hasil scraping:

- diklasifikasikan oleh KNN
- diklasifikasikan oleh SVM
- digabungkan dengan `Soft Voting`
- bisa dilihat dalam tabel
- bisa ditampilkan dalam dashboard
- bisa diunduh sebagai CSV

Untuk rentang tanggal yang terlalu panjang, situs web akan membuat penyimpanan sementara ke database dan scraping dapat dilanjutkan kapan saja.

### 4. Riwayat Aktivitas

Riwayat dipisah menjadi:

- `Riwayat Scraping`
- `Riwayat Prediksi`

### 5. Admin Panel

Route:

- `/admin/`

Situs khusus untuk Administrator sehingga Administrator dapat melakukan:

- tambah user
- edit user
- ubah password user
- atur `is_staff`, `is_superuser`, `is_active`
- hapus user
- lihat dataset `PredictionHistory`
- lihat dataset `ScrapeHistory`
- edit dan hapus dataset history

## Output Model

Setiap hasil klasifikasi menyimpan tiga keluaran:

- `KNN`
- `SVM`
- `Soft Voting`

### KNN

Disimpan sebagai:

- `Probabilitas Positif KNN`
- `Probabilitas Negatif KNN`
- label `KNN`

Skor KNN diambil dari `predict_proba` kelas positif, lalu probabilitas negatif dihitung dari pasangan kelasnya.

### SVM

Disimpan sebagai:

- `Probabilitas Positif SVM`
- `Probabilitas Negatif SVM`
- label `SVM`

Skor SVM tidak memakai `predict_proba` native. Nilainya berasal dari:

1. `decision_function`
2. diubah dengan rumus sigmoid
3. diperlakukan sebagai probabilitas positif `0-1`

### Soft Voting

Disimpan sebagai:

- `Probabilitas Positif Soft Voting`
- `Probabilitas Negatif Soft Voting`
- label `Soft Voting`

Bobot saat ini seimbang yaitu:

- KNN: `0.5`
- SVM: `0.5`

## Aturan Label Netral

Threshold label netral saat ini:

- KNN: `0.4 - 0.6`
- SVM: `0.4 - 0.6` setelah sigmoid
- Soft Voting: `0.4 - 0.6`

## Ekspor CSV

Aplikasi mendukung unduh CSV untuk:

- hasil prediksi file (`PredictionHistory`)
- hasil klasifikasi scraping (`ScrapeHistory`)

## Preprocessing

Preprocessing utama ada di:

- [model_service.py](sentiment_app/services/model_service.py)

File pendukung preprocessing:
- `sentiment_site/models/stopwords-id.txt`
- `sentiment_site/models/stopwords-id(wordcloud).txt` untuk WordCloud
- `sentiment_site/models/singkatan.tsv`

## Struktur Model

Direktori model default:

```text
sentiment_site/models/
```

Artifact yang didukung:

```text
knn_model.joblib
svm_linear_model.joblib
svm_rbf_model.joblib
vectorizer.joblib
tfidf_vectorizer.joblib
label_encoder.joblib
stopwords-id.txt
singkatan.tsv
```

Catatan:
- `knn_model.joblib` dan `svm_linear_model.joblib` wajib ada
- `vectorizer.joblib` atau `tfidf_vectorizer.joblib` dibutuhkan jika model bukan pipeline end-to-end

## Instalasi Lokal

### 1. Clone repository

```bash
git clone <repo-url>
cd Web
```

### 2. Buat virtual environment

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Bash:

```bash
source .venv/bin/activate
```

### 3. Install dependency

```bash
pip install -r requirements.txt
```

### 4. Siapkan environment

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

### 5. Jalankan migrasi

```bash
python manage.py migrate
```

Jika perlu akun admin:

```bash
python manage.py createsuperuser
```

### 6. Jalankan server

```bash
python manage.py runserver
```

Default local URL:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/admin/`

## Testing

Jalankan semua test:

```bash
python manage.py test
```

Cek konfigurasi cepat:

```bash
python manage.py check
```

## Keamanan

- validasi upload extension, type, dan size
- CSRF aktif
- validasi path aman untuk file unduhan
- API key scraping tidak disimpan ke database
- password user disimpan sebagai hash oleh Django

## Troubleshooting

- `No module named 'django'`:
  aktifkan virtual environment dulu atau jalankan dengan `.\.venv\Scripts\python.exe`
- model gagal dimuat:
  periksa file di `sentiment_site/models/`
- scraping lambat:
  kecilkan rentang tanggal atau sesuaikan limit tweet di `.env`
- wordcloud tidak muncul:
  pastikan dependency dan file stopword tersedia, atau cek batas `SENTIMENT_WORDCLOUD_MAX_ROWS`
