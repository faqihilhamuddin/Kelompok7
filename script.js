const signInButton = document.getElementById('signInButton');
const signUpButton = document.getElementById('signUpButton');
const logout = document.getElementById('logout');
const signInForm = document.getElementById('signInForm');
const signUpForm = document.getElementById('signUpForm');

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

//ke Sign Up
signUpButton.addEventListener('click', function() {
    signUpForm.style.display = "block";
    signInForm.style.display = "none";
    signUpButton.classList.add('active');
    signInButton.classList.remove('active');
});

signUpButton.addEventListener('click', function() {
    signUpForm.style.display = "block";
    signInForm.style.display = "none";
    signUpButton.classList.add('active');
    signInButton.classList.remove('active');
});
