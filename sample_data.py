def get_sample_data():
    """
    Get sample Indonesian passenger complaints for demonstration
    
    Returns:
        list: List of sample complaint texts
    """
    return [
        "Bus selalu terlambat dan sopir tidak ramah saat saya bertanya",
        "AC di dalam kereta tidak berfungsi, sangat panas dan tidak nyaman",
        "Tiket online yang saya beli tidak bisa digunakan, tapi uang sudah terpotong",
        "Bagasi saya hilang setelah penerbangan dan staf tidak membantu sama sekali",
        "Kursi di bus sangat kotor dan berbau tidak sedap, tidak layak digunakan",
        "Penerbangan delay 5 jam tanpa ada informasi yang jelas dari maskapai",
        "Sopir taksi menaikkan argo secara tidak wajar dan meminta tip yang besar",
        "Toilet di terminal sangat kotor dan tidak ada air, perlu perawatan segera",
        "Informasi jadwal di stasiun tidak akurat, membuat saya ketinggalan kereta",
        "Pelayanan customer service sangat lambat, saya menunggu 1 jam untuk dilayani",
        "Harga tiket naik tiba-tiba tanpa pemberitahuan sebelumnya, sangat tidak transparan",
        "WiFi di bandara tidak berfungsi padahal sudah diiklankan gratis dan cepat",
        "Jumlah kursi di ruang tunggu sangat terbatas, banyak penumpang yang berdiri",
        "Makanan di pesawat tidak layak, nasi dingin dan lauk sudah berbau",
        "Sistem check-in online sering error, terpaksa harus datang lebih awal ke bandara",
        "Terminal bus sangat tidak aman, barang bawaan saya dicuri saat saya tertidur",
        "Penumpang merokok di dalam kereta tapi petugas tidak menegur sama sekali",
        "Petunjuk arah di bandara membingungkan, hampir terlambat check-in pesawat",
        "Supir bus ngebut dan mengemudi secara tidak aman, membahayakan penumpang",
        "Pintu kereta rusak dan tidak bisa ditutup dengan benar selama perjalanan"
    ]

def get_sample_data_with_labels():
    """
    Get sample Indonesian passenger complaints with category labels for training
    
    Returns:
        list: List of tuples containing (complaint_text, category)
    """
    return [
        # Layanan category
        ("Bus selalu terlambat dan sopir tidak ramah saat saya bertanya", "Layanan"),
        ("Pelayanan customer service sangat lambat, saya menunggu 1 jam untuk dilayani", "Layanan"),
        ("Sopir taksi menaikkan argo secara tidak wajar dan meminta tip yang besar", "Layanan"),
        ("Petugas loket tidak ramah saat melayani pembelian tiket", "Layanan"),
        ("Sopir bus tidak membantu penumpang lansia yang kesulitan naik", "Layanan"),
        ("Staf maskapai tidak memberikan informasi yang jelas tentang delay", "Layanan"),
        ("Pramugari di pesawat tidak merespon ketika saya meminta bantuan", "Layanan"),
        ("Petugas di stasiun tidak bisa menjawab pertanyaan tentang jadwal kereta", "Layanan"),
        ("Pelayanan bagasi sangat lambat, harus menunggu 2 jam untuk mendapatkan koper", "Layanan"),
        ("Petugas keamanan di bandara bersikap kasar saat pemeriksaan", "Layanan"),
        
        # Fasilitas category
        ("AC di dalam kereta tidak berfungsi, sangat panas dan tidak nyaman", "Fasilitas"),
        ("Kursi di bus sangat kotor dan berbau tidak sedap, tidak layak digunakan", "Fasilitas"),
        ("Toilet di terminal sangat kotor dan tidak ada air, perlu perawatan segera", "Fasilitas"),
        ("WiFi di bandara tidak berfungsi padahal sudah diiklankan gratis dan cepat", "Fasilitas"),
        ("Jumlah kursi di ruang tunggu sangat terbatas, banyak penumpang yang berdiri", "Fasilitas"),
        ("Eskalator di stasiun kereta rusak sejak bulan lalu dan belum diperbaiki", "Fasilitas"),
        ("Tidak ada pendingin udara di ruang tunggu bandara, sangat tidak nyaman", "Fasilitas"),
        ("Tempat duduk di ferry rusak dan beberapa tidak bisa digunakan", "Fasilitas"),
        ("Tidak ada fasilitas untuk penyandang disabilitas di terminal bus", "Fasilitas"),
        ("Lampu di dalam gerbong kereta mati, perjalanan dalam kegelapan", "Fasilitas"),
        
        # Teknis category
        ("Tiket online yang saya beli tidak bisa digunakan, tapi uang sudah terpotong", "Teknis"),
        ("Penerbangan delay 5 jam tanpa ada informasi yang jelas dari maskapai", "Teknis"),
        ("Informasi jadwal di stasiun tidak akurat, membuat saya ketinggalan kereta", "Teknis"),
        ("Sistem check-in online sering error, terpaksa harus datang lebih awal ke bandara", "Teknis"),
        ("Pintu kereta rusak dan tidak bisa ditutup dengan benar selama perjalanan", "Teknis"),
        ("Mesin tiket otomatis di stasiun tidak berfungsi dengan baik", "Teknis"),
        ("Website pemesanan tiket down saat peak season liburan", "Teknis"),
        ("Jadwal keberangkatan selalu tidak tepat waktu setiap hari", "Teknis"),
        ("Sistem boarding pass elektronik error saat akan masuk ke pesawat", "Teknis"),
        ("Pembayaran dengan kartu debit tidak bisa diproses di loket tiket", "Teknis"),
        
        # Keamanan category
        ("Bagasi saya hilang setelah penerbangan dan staf tidak membantu sama sekali", "Keamanan"),
        ("Terminal bus sangat tidak aman, barang bawaan saya dicuri saat saya tertidur", "Keamanan"),
        ("Supir bus ngebut dan mengemudi secara tidak aman, membahayakan penumpang", "Keamanan"),
        ("Tidak ada pemeriksaan keamanan yang ketat di terminal ferry", "Keamanan"),
        ("Penumpang membawa barang berbahaya di dalam kereta tanpa diperiksa", "Keamanan"),
        ("Keamanan di area parkir sangat minim, mobil saya tergores", "Keamanan"),
        ("Tidak ada petugas keamanan di peron stasiun pada malam hari", "Keamanan"),
        ("Sopir bus menggunakan handphone saat mengemudi, sangat berbahaya", "Keamanan"),
        ("Pintu darurat di pesawat tampak rusak dan tidak bisa dibuka", "Keamanan"),
        ("Tidak ada briefing keselamatan sebelum kapal berangkat", "Keamanan"),
        
        # Harga category
        ("Harga tiket naik tiba-tiba tanpa pemberitahuan sebelumnya, sangat tidak transparan", "Harga"),
        ("Tarif bagasi terlalu mahal dibandingkan dengan maskapai lain", "Harga"),
        ("Harga makanan di kereta sangat tidak masuk akal, terlalu mahal", "Harga"),
        ("Biaya tambahan yang tidak dijelaskan saat pembelian tiket online", "Harga"),
        ("Tarif taksi bandara tidak sesuai dengan estimasi yang diberikan", "Harga"),
        ("Diskon yang diiklankan tidak berlaku saat saya membeli tiket", "Harga"),
        ("Harga tiket pulang pergi lebih mahal daripada membeli tiket terpisah", "Harga"),
        ("Tidak ada pengembalian dana untuk pembatalan tiket mendadak", "Harga"),
        ("Harga tiket untuk tujuan yang sama bervariasi tanpa alasan yang jelas", "Harga"),
        ("Biaya reschedule tiket sangat tinggi meskipun dilakukan jauh hari", "Harga"),
        
        # Kenyamanan category
        ("Makanan di pesawat tidak layak, nasi dingin dan lauk sudah berbau", "Kenyamanan"),
        ("Penumpang merokok di dalam kereta tapi petugas tidak menegur sama sekali", "Kenyamanan"),
        ("Petunjuk arah di bandara membingungkan, hampir terlambat check-in pesawat", "Kenyamanan"),
        ("Jarak antar kursi di bus terlalu sempit, kaki tidak bisa diluruskan", "Kenyamanan"),
        ("Suhu di dalam pesawat terlalu dingin, tidak ada selimut yang disediakan", "Kenyamanan"),
        ("Kebisingan dari mesin kereta sangat mengganggu, tidak bisa istirahat", "Kenyamanan"),
        ("Tidak ada hiburan selama penerbangan panjang, sangat membosankan", "Kenyamanan"),
        ("Kursi tidak bisa direbahkan padahal perjalanan sangat jauh", "Kenyamanan"),
        ("Terlalu banyak pengumuman melalui speaker selama perjalanan malam", "Kenyamanan"),
        ("Bau tidak sedap dari toilet kereta tercium sampai ke gerbong", "Kenyamanan")
    ]
