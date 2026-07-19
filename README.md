# Sistem Analisis Sentimen (KNN + SVM)

Aplikasi web Django untuk analisis sentimen Bahasa Indonesia menggunakan model KNN dan SVM, dengan hasil gabungan `Soft Voting`.

Website:

- `https://analisissentimen.up.railway.app/`

## Ringkasan

Fitur utama aplikasi:

- login wajib untuk semua fitur utama
- analisis sentimen dari `CSV/TXT`
- pengumpulan data X melalui `twitterapi.io`
- pemilihan versi model sentimen
- riwayat per user untuk `Pengumpulan Data X` dan `CSV/TXT`
- tampilan `Tabel` dan `Dashboard` untuk riwayat
- unduh hasil klasifikasi dalam format `CSV`, `XLSX`, dan `JPG` (dashboard)

## Halaman Utama

### 1. Beranda

Berisi ringkasan:

- total pengumpulan data X
- total tweet terkumpul
- total riwayat CSV/TXT
- total data hasil CSV/TXT
- panduan cara menggunakan menu utama

### 2. Buat Analisis

Mendukung dua mode:

- `Pengumpulan Data X`
- `CSV/TXT`

Tab `Pengumpulan Data X` digunakan untuk:

- memilih versi model
- memasukkan API key `twitterapi.io`
- memasukkan kueri pencarian
- memilih rentang tanggal
- menjalankan pengambilan tweet lalu langsung klasifikasi sentimen

Tab `CSV/TXT` digunakan untuk:

- memilih versi model
- unggah file `CSV` atau `TXT`
- memilih kolom teks jika diperlukan
- menjalankan klasifikasi sentimen dari file

### 3. Riwayat Aktivitas

Riwayat dipisah menjadi:

- `Riwayat Pengumpulan Data X`
- `Riwayat CSV/TXT`

Setiap riwayat dapat dibuka dalam dua mode:

- `Tabel`
- `Dashboard`

Riwayat pengumpulan data X yang belum selesai dapat dilanjutkan kembali dari halaman detail riwayat.

### 4. Admin Panel

Route:

- `/admin/`

Situs khusus untuk Administrator sehingga Administrator dapat melakukan:

- tambah user
- edit user
- ubah password user
- atur peran user dan status aktif
- hapus user
- tambah versi model
- edit versi model
- hapus versi model
- lihat database `Hasil Pengumpulan Data X`
- lihat database `Hasil CSV/TXT`
- hapus dataset history

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

## Ekspor Hasil

Aplikasi mendukung unduh hasil untuk:

- hasil `CSV/TXT` dalam format `CSV` dan `XLSX`
- hasil `Pengumpulan Data X` dalam format `CSV` dan `XLSX`
- dashboard riwayat dalam format `JPG`

## Preprocessing

Preprocessing utama ada di:

- [model_service.py](sentiment_app/services/model_service.py)

File pendukung preprocessing:
- `SENTIMENT_MODELS_DIR/stopwords-id.txt`
- `SENTIMENT_MODELS_DIR/stopwords-id(wordcloud).txt` untuk WordCloud
- `SENTIMENT_MODELS_DIR/singkatan.tsv`

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
- API key `twitterapi.io` tidak disimpan ke database
- password user disimpan sebagai hash oleh Django

## Troubleshooting

- `No module named 'django'`:
  aktifkan virtual environment dulu atau jalankan dengan `.\.venv\Scripts\python.exe`
- model gagal dimuat:
  periksa file upload model di storage Django atau asset fallback di `SENTIMENT_MODELS_DIR`
- pengumpulan data X lambat:
  kecilkan rentang tanggal atau sesuaikan limit tweet di `.env`
- wordcloud tidak muncul:
  pastikan dependency dan file stopword tersedia, atau cek batas `SENTIMENT_WORDCLOUD_MAX_ROWS`
