import streamlit_authenticator as stauth
import streamlit as st


def get_authenticator():
    credentials = {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Admin User',
                # Password: admin123
                'password': '$2b$12$X1c4dHAOSx.4JGrqWwcNhemv4ELzKOtt44awFMBE1kT.wdpd0nMm6'
            },
            'subramani': {
                'email': 'subramani@example.com',
                'name': 'Subramani BS',
                # Password: subramani123
                'password': '$2b$12$OHc3jKshuQZpRmxSOfN6We8I1VqI7vtD0YhpFDbGYa9X2vEfalEj6'
            },
            'user1': {
                'email': 'user1@example.com',
                'name': 'User One',
                # Password: user123
                'password': '$2b$12$Cni.62kdvE6xUnSAvZKjIu2q.R9cEODi4ZbHt52g67oWVLfqq.P2G'
            }
        }
    }

    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name='ai_sql_analyst',
        cookie_key='super_secret_key_123',
        cookie_expiry_days=7
    )

    return authenticator


def show_login(authenticator):
    st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #e6edf3; }
        .login-box {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 30px;
            max-width: 400px;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='color:#00b4d8;'>🧠 AI SQL Analyst</h1>
            <p style='color:#aaa;'>Please login to continue</p>
        </div>
        """, unsafe_allow_html=True)

        authenticator.login(location='main')