# voetbalrecsys
Tempat menyimpan file terkait tugas besar mata kuliah Recommender System

# Link Aplikasi Streamlit : https://voetbalrecsys-kelompok09.streamlit.app/

# Source dataset : https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Kelompok 9 
Bayu Seno Nugroho - 1301213270

Abidzar Ahmad Haikal - 1301213288

Satya Rayyis Baruna - 1301213316

Muhammad Rayhan Saniputra - 1301213262

Deskripsi

  Program ini merupakan implementasi sistem rekomendasi berbasis konten (Content-Based Recommender System) untuk pertandingan sepak bola internasional yang menganalisis dataset komprehensif berisi 47,917 hasil pertandingan dari tahun 1872 hingga 2024. Program menggunakan tiga dataset utama: results.csv yang berisi informasi detail pertandingan seperti tanggal, tim, skor, turnamen, dan lokasi; goalscorers.csv yang mencatat data pencetak gol termasuk informasi penalti dan gol bunuh diri; serta shootouts.csv yang menyimpan data adu penalti. Sistem ini dibangun menggunakan framework Streamlit untuk membuat dashboard interaktif, dengan fitur utama berupa rekomendasi pertandingan berdasarkan kemiripan karakteristik tim menggunakan teknik cosine similarity, di mana sistem mengekstrak berbagai fitur dari setiap tim seperti rata-rata gol yang dicetak, win rate, dan jumlah pertandingan yang dimainkan.

  Dashboard yang dikembangkan menyajikan berbagai fitur analisis yang komprehensif, dimulai dari ekstraksi fitur tim seperti rata-rata gol yang dicetak (baik sebagai tuan rumah maupun tamu), win rate, dan jumlah pertandingan. Program mengimplementasikan preprocessing data untuk mengubah format tanggal dan menambahkan kolom tahun, serta menghitung berbagai metrik performa tim. Sistem rekomendasi menggunakan MinMaxScaler untuk normalisasi fitur dan cosine similarity untuk menghitung kemiripan antar tim, yang kemudian digunakan untuk merekomendasikan pertandingan berdasarkan tim-tim yang memiliki karakteristik serupa.

  Komponen interaktif dashboard mencakup peta lokasi pertandingan yang menampilkan persebaran geografis, analisis gol yang memvisualisasikan tren dan statistik pencetak gol, serta analisis head-to-head antar tim. Pengguna dapat menggunakan berbagai filter di sidebar untuk mengeksplorasi data berdasarkan tahun (1872-2024), tim, dan turnamen. Program juga menyertakan evaluasi performa menggunakan metrik MSE dan RMSE untuk mengukur akurasi rekomendasi. Visualisasi data menggunakan Plotly Express menghasilkan grafik yang informatif, termasuk tren gol sepanjang waktu, statistik top scorer, dan analisis situasi gol (penalti dan gol bunuh diri). Keseluruhan sistem dirancang untuk memberikan pemahaman mendalam tentang pola dan tren dalam sepak bola internasional, serta membantu pengguna menemukan pertandingan yang mungkin menarik berdasarkan preferensi tim mereka.
