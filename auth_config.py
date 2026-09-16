import bcrypt

def hash_password(password):
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

# Generate hashes for your passwords
passwords = {
    'admin': 'admin123',
    'subramani': 'subramani123',
    'user1': 'user123'
}

for username, password in passwords.items():
    hashed = hash_password(password)
    print(f"{username}: {hashed}")