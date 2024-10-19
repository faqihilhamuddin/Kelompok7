const signInButton = document.getElementById('signInButton');  // ID tombol Masuk
const signUpButton = document.getElementById('signUpButton');  // ID tombol Daftar
const submitSignIn = document.getElementById('submitSignIn');  // Tombol submit Masuk
const submitSignUp = document.getElementById('submitSignUp');  // Tombol submit Daftar
const signInForm = document.getElementById('signInForm');  // Form Masuk
const signUpForm = document.getElementById('signUpForm');  // Form Daftar
const signInMessage = document.getElementById('signInMessage'); // Pesan kesalahan saat login
const signUpMessage = document.getElementById('signUpMessage'); // Pesan kesalahan saat daftar

// Username dan password admin
const adminUsername = 'admin';
const adminPassword = 'admin';

// Tampilan awal: Tampilkan form masuk dan sembunyikan form daftar
signInForm.style.display = "block";
signUpForm.style.display = "none";

// Saat klik tombol "Masuk", tampilkan form masuk dan sembunyikan form daftar
signInButton.addEventListener('click', function() {
    signInForm.style.display = "block";
    signUpForm.style.display = "none";
    signInButton.classList.add('active');
    signUpButton.classList.remove('active');
});

// Saat klik tombol "Daftar", tampilkan form daftar dan sembunyikan form masuk
signUpButton.addEventListener('click', function() {
    signUpForm.style.display = "block";
    signInForm.style.display = "none";
    signUpButton.classList.add('active');
    signInButton.classList.remove('active');
});

// Event listener untuk tombol submit "Masuk"
submitSignIn.addEventListener('click', function() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const role = document.getElementById('roleSelect').value;

    // Validasi input
    if (!email || !password || !role) {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Semua kolom harus diisi!';
        return;
    }

    // Cek login admin
    if (role === 'admin' && email === adminUsername && password === adminPassword) {
        window.location.href = 'admhome.html'; // Redirect ke halaman admin
    } else if (role === 'user') {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Login berhasil untuk user!';
    } else {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Email, password, atau role tidak valid!';
    }
});

// Event listener untuk tombol submit "Daftar"
submitSignUp.addEventListener('click', function() {
    const fName = document.getElementById('fName').value;
    const lName = document.getElementById('lName').value;
    const rEmail = document.getElementById('rEmail').value;
    const rPassword = document.getElementById('rPassword').value;

    // Validasi input
    if (!fName || !lName || !rEmail || !rPassword) {
        signUpMessage.style.display = 'block';
        signUpMessage.innerHTML = 'Semua kolom harus diisi!';
        return;
    }

    // Tampilkan pesan sukses (dalam skenario ini, tidak ada penyimpanan data)
    signUpMessage.style.display = 'block';
    signUpMessage.innerHTML = 'Pendaftaran berhasil!';
});
