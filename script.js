const signInButton = document.getElementById('submitSignIn');  // Pastikan ID benar
const signUpButton = document.getElementById('submitSignUp');  // Pastikan ID benar
const logout = document.getElementById('logout');
const signInForm = document.getElementById('signInForm');
const signUpForm = document.getElementById('signUpForm');
const signInMessage = document.getElementById('signInMessage'); // Pesan kesalahan

// Username dan password admin
const adminUsername = 'admin';
const adminPassword = 'admin';

// Tampilan awal
signInForm.style.display = "block";
signInForm.classList.add('active');
signInButton.classList.add('active');

// ke Sign In
signInButton.addEventListener('click', function() {
    signInForm.style.display = "block";
    signUpForm.style.display = "none";
    signInButton.classList.add('active');
    signUpButton.classList.remove('active');
});

// ke Sign Up
signUpButton.addEventListener('click', function() {
    signUpForm.style.display = "block";
    signInForm.style.display = "none";
    signUpButton.classList.add('active');
    signInButton.classList.remove('active');
});

// login
signInButton.addEventListener('click', function() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const role = document.getElementById('roleSelect').value;

    // Validasi input
    if (!email || !password || !role) {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Semua kolom harus diisi!';
        return;
    }

    
    if (role === 'admin' && email === adminUsername && password === adminPassword) {
        window.location.href = 'admhome.html'; // Arahkan ke halaman admin
    } else if (role === 'user') {
        
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Fitur login pengguna sedang dalam pengembangan.';
    } else {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Email, password, atau role tidak valid!';
    }
});

// Logout button functionality (jika diperlukan)
if (logout) {
    logout.addEventListener('click', function() {
        
        signInForm.style.display = "block";
        signUpForm.style.display = "none";
        signInButton.classList.add('active');
        signUpButton.classList.remove('active');
        
       
    });
};
