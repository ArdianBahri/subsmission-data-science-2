# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Bagian ini menjelaskan latar belakang, permasalahan, dan tujuan dari proyek prediksi kesuksesan mahasiswa untuk perusahaan edutech "Jaya Jaya Institut".

### Latar Belakang Bisnis
Jaya Jaya Institut adalah sebuah institusi pendidikan yang sedang menghadapi tantangan bisnis berupa tingginya tingkat *dropout* mahasiswa, yang saat ini berada pada angka 32%. Tingkat *dropout* yang tinggi ini tidak hanya berdampak pada pendapatan institusi tetapi juga pada reputasi akademik. Oleh karena itu, perusahaan berinisiatif menggunakan pendekatan *data science* untuk mengidentifikasi mahasiswa yang berisiko *dropout* secara dini, sehingga intervensi yang tepat dapat dilakukan untuk meningkatkan tingkat retensi (*retention rate*).

### Permasalahan Bisnis
Berdasarkan latar belakang tersebut, permasalahan bisnis utama yang akan diselesaikan adalah:
* **Prediksi Hasil Mahasiswa:** Mengembangkan model yang dapat secara akurat memprediksi hasil akhir mahasiswa, apakah mereka akan Lulus (*Graduate*), *Dropout*, atau masih Terdaftar (*Enrolled*).
* **Identifikasi Faktor Risiko:** Mengidentifikasi faktor-faktor kunci (baik akademik, demografis, maupun finansial) yang paling signifikan mempengaruhi keberhasilan atau kegagalan mahasiswa.
* **Strategi Retensi:** Memberikan *insight* berbasis data yang dapat digunakan untuk merancang dan mengimplementasikan strategi retensi yang efektif dan tepat sasaran.
* **Peningkatan ROI:** Mencapai target bisnis untuk mengurangi angka *dropout* sebesar 15-25% yang diproyeksikan dapat meningkatkan pendapatan hingga $2M per tahun.

### Cakupan Proyek
Proyek ini mencakup implementasi lengkap metodologi CRISP-DM, mulai dari pemahaman bisnis hingga deployment. Cakupan utamanya adalah:
* **Analisis Data Eksploratif (EDA):** Melakukan analisis mendalam terhadap data historis mahasiswa untuk memahami pola dan tren yang ada.
* ***Feature Engineering*:** Membuat fitur-fitur baru yang relevan seperti `academic_risk` dan `financial_risk` untuk meningkatkan performa model.
* **Pengembangan Model *Machine Learning*:** Membangun dan mengevaluasi beberapa algoritma klasifikasi, lalu melakukan optimisasi pada model terbaik untuk memprediksi status mahasiswa.
* **Pengembangan *Business Dashboard*:** Membuat *dashboard business intelligence* menggunakan Metabase untuk memonitor KPI dan hasil analisis secara interaktif.
* **Pengembangan Prototipe Sistem:** Membuat aplikasi web interaktif menggunakan Streamlit sebagai prototipe sistem peringatan dini (*early warning system*) yang dapat digunakan oleh staf akademik.
* **Rekomendasi Bisnis:** Memberikan rekomendasi strategis yang dapat ditindaklanjuti oleh manajemen berdasarkan hasil analisis dan evaluasi model.

### Persiapan

**Sumber Data**:
Sumber data: [students_performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

**Setup Environment**:
```
python -m venv venv
venv\Scripts\activate
```

**Menginstall library yang diperlukan:**
```
pip install -r requirements.txt # install requirements.txt untuk aplikasi streamlit
```

**Menjalankan model menggunakan streamlit:**
```
streamlit run apps.py
```

## Business Dashboard
Sebuah dashboard business intelligence (BI) telah dikembangkan menggunakan **Metabase** untuk memberikan wawasan visual kepada para pemangku kepentingan (manajemen, staf akademik, dll.). Dashboard ini terhubung ke database PostgreSQL yang berisi data prediksi dan analisis.

Fitur utama dari dashboard ini meliputi:
-   **Student Risk Overview**: Visualisasi distribusi mahasiswa berdasarkan tingkat risiko (*dropout*) secara *real-time*.
-   **Performance Metrics**: Pemantauan metrik performa model seperti akurasi, presisi, dan *recall*.
-   **Demographic Analysis**: Analisis *breakdown* risiko mahasiswa berdasarkan demografi (usia, gender, dll).
-   **Academic Trends**: Analisis tren performa akademik dan faktor-faktor yang mempengaruhinya.
-   **Economic Impact**: Analisis dampak faktor eksternal (inflasi, PDB) terhadap kelulusan mahasiswa.

Dashboard dapat diakses melalui Docker deployment pada link berikut: `http://localhost:3000`

**Akun Metabase:**
```bash
Email: ardianbahri20@gmail.com
Password: Ardian404
```

## Menjalankan Sistem Machine Learning
Prototipe sistem prediksi telah dibuat dalam bentuk aplikasi web interaktif menggunakan **Streamlit**. Aplikasi ini memungkinkan pengguna untuk:

-   **Prediksi Individual**: Memasukkan data seorang mahasiswa melalui form dan mendapatkan hasil prediksi risiko *dropout* secara *real-time*.
-   **Prediksi Batch**: Mengunggah file CSV yang berisi data beberapa mahasiswa untuk diproses secara bersamaan.

Cara paling mudah untuk menjalankan keseluruhan sistem (Aplikasi Web, Database, dan Dashboard) adalah menggunakan Docker.

```bash
# Clone repository
git clone https://github.com/ArdianBahri/subsmission-data-science-2.git
cd student-success-prediction

# Jalankan seluruh stack
docker-compose up -d
```

jika ingin menjalankan secara lokal di komputer
```
streamlit run apps.py
```

**jika ingin menjalankan langsung dengan mengunjungi website**
Untuk menjalankan sistem, silahkan kunjungi :
Prototipe sistem machine learning dapat diakses pada link Streamlit: 
```bash 
https://datasice-edutech.streamlit.app/
```
## Conclusion
Proyek ini berhasil mengembangkan sebuah sistem untuk memprediksi status kelulusan mahasiswa di Jaya Jaya Institut menggunakan model *machine learning*. Berdasarkan hasil perbandingan beberapa algoritma, model **Random Forest** terpilih sebagai model terbaik dengan performa yang cukup baik pada data uji.

-   **Akurasi Keseluruhan**: 78%
-   **F1-Score (Macro Avg)**: 0.70
-   **Recall (untuk kelas Dropout)**: 77.1%, yang berarti model berhasil mengidentifikasi 77% dari seluruh mahasiswa yang sebenarnya *dropout*.
-   **Precision (untuk kelas Dropout)**: 81.1%, yang berarti dari semua mahasiswa yang diprediksi *dropout*, 81% di antaranya benar-benar *dropout*.

Analisis fitur menunjukkan bahwa faktor penentu utama seorang mahasiswa berisiko *dropout* adalah performa akademik di semester-semester awal (tingkat keberhasilan akademik dan nilai semester) serta kondisi finansial (status pembayaran SPP dan status beasiswa).

### Rekomendasi Action Items
Berdasarkan temuan dari analisis dan performa model, berikut adalah beberapa rekomendasi yang dapat ditindaklanjuti oleh perusahaan:

-   **Implementasikan Sistem Peringatan Dini**: Terapkan model ini sebagai sistem peringatan dini (*early warning system*). Mulai dengan program percontohan (*pilot program*) di satu atau dua departemen untuk memvalidasi efektivitasnya dalam lingkungan nyata sebelum diterapkan di seluruh institusi.
-   **Kembangkan Program Intervensi Terarah**: Buat program intervensi yang menargetkan mahasiswa berisiko tinggi berdasarkan faktor pendorongnya. Contohnya, mahasiswa dengan `academic_risk` tinggi diberikan bimbingan akademik tambahan, sementara yang memiliki `financial_risk` tinggi dapat diarahkan ke program bantuan keuangan atau beasiswa.
-   **Optimalkan Alokasi Sumber Daya**: Gunakan hasil prediksi untuk mengalokasikan sumber daya (seperti konselor akademik, tutor, atau dana bantuan) secara lebih efisien kepada mahasiswa yang paling membutuhkannya, sehingga dapat memaksimalkan dampak retensi dan ROI.

