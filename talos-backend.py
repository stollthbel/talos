# talos_backend.py — Phase 4/6 Backend File (Flask API w/ Stripe + ZK + User Profiles)
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import jwt
import datetime
import stripe
import hashlib
import os
import json

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key'

# Stripe configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_your_key_here')
STRIPE_ENDPOINT_SECRET = os.getenv('STRIPE_ENDPOINT_SECRET', 'whsec_your_endpoint_secret_here')

# Initialize database
def init_db():
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            subscription_status TEXT DEFAULT 'free',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS zk_commitments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            commitment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            pnl_percent REAL,
            outcome TEXT,
            status TEXT DEFAULT 'Closed',
            symbol TEXT,
            type TEXT,
            amount REAL,
            price REAL,
            notes TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            title TEXT,
            content TEXT,
            mood TEXT DEFAULT 'neutral',
            market_conditions TEXT,
            lessons TEXT,
            next_steps TEXT,
            tags TEXT,
            drawings TEXT,
            is_private BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            symbol TEXT,
            quantity REAL,
            avg_price REAL,
            current_price REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            price REAL,
            volume REAL,
            change_percent REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Journal Endpoints ---
@app.route('/api/journal_entries', methods=['GET'])
def get_journal_entries():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, title, content, mood, market_conditions, lessons, next_steps, 
               tags, drawings, is_private, created_at, updated_at
        FROM journal_entries 
        WHERE username = ? 
        ORDER BY created_at DESC
    ''', (username,))
    
    entries = []
    for row in c.fetchall():
        entries.append({
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'mood': row[3],
            'marketConditions': row[4],
            'lessons': row[5],
            'nextSteps': row[6],
            'tags': row[7].split(',') if row[7] else [],
            'drawings': json.loads(row[8]) if row[8] else [],
            'isPrivate': bool(row[9]),
            'createdAt': row[10],
            'updatedAt': row[11]
        })
    
    conn.close()
    return jsonify(entries)

@app.route('/api/journal_entries', methods=['POST'])
def create_journal_entry():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO journal_entries 
        (username, title, content, mood, market_conditions, lessons, next_steps, tags, drawings, is_private)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        data.get('title', ''),
        data.get('content', ''),
        data.get('mood', 'neutral'),
        data.get('marketConditions', ''),
        data.get('lessons', ''),
        data.get('nextSteps', ''),
        ','.join(data.get('tags', [])),
        json.dumps(data.get('drawings', [])),
        data.get('isPrivate', False)
    ))
    
    entry_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({"id": entry_id, "message": "Journal entry created successfully"})

@app.route('/api/journal_entries/<int:entry_id>', methods=['PUT'])
def update_journal_entry(entry_id):
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    
    c.execute('''
        UPDATE journal_entries 
        SET title = ?, content = ?, mood = ?, market_conditions = ?, lessons = ?, 
            next_steps = ?, tags = ?, drawings = ?, is_private = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND username = ?
    ''', (
        data.get('title', ''),
        data.get('content', ''),
        data.get('mood', 'neutral'),
        data.get('marketConditions', ''),
        data.get('lessons', ''),
        data.get('nextSteps', ''),
        ','.join(data.get('tags', [])),
        json.dumps(data.get('drawings', [])),
        data.get('isPrivate', False),
        entry_id,
        username
    ))
    
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Journal entry updated successfully"})

@app.route('/api/journal_entries/<int:entry_id>', methods=['DELETE'])
def delete_journal_entry(entry_id):
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('DELETE FROM journal_entries WHERE id = ? AND username = ?', (entry_id, username))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Journal entry deleted successfully"})

# --- Enhanced Trade Endpoints ---
@app.route('/api/trades', methods=['GET'])
def get_trades():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, symbol, type, amount, price, pnl_percent, outcome, status, notes
        FROM trades 
        WHERE username = ? 
        ORDER BY timestamp DESC
    ''', (username,))
    
    trades = []
    for row in c.fetchall():
        trades.append({
            'id': row[0],
            'date': row[1],
            'symbol': row[2],
            'type': row[3],
            'amount': row[4],
            'price': row[5],
            'pnl': row[6],
            'outcome': row[7],
            'status': row[8],
            'notes': row[9]
        })
    
    conn.close()
    return jsonify(trades)

# --- Portfolio Endpoints ---
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock portfolio data
    portfolio = [
        {'symbol': 'AAPL', 'quantity': 50, 'avgPrice': 150.00, 'currentPrice': 175.00, 'pnl': 1250.00},
        {'symbol': 'GOOGL', 'quantity': 10, 'avgPrice': 2800.00, 'currentPrice': 2900.00, 'pnl': 1000.00},
        {'symbol': 'TSLA', 'quantity': 25, 'avgPrice': 800.00, 'currentPrice': 780.00, 'pnl': -500.00}
    ]
    
    return jsonify(portfolio)

# --- Market Data Endpoints ---
@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock market data
    market_data = [
        {'symbol': 'SPY', 'price': 430.50, 'change': 2.5, 'volume': 50000000},
        {'symbol': 'QQQ', 'price': 350.25, 'change': -1.2, 'volume': 30000000},
        {'symbol': 'VTI', 'price': 225.75, 'change': 0.8, 'volume': 25000000}
    ]
    
    return jsonify(market_data)

# --- Analytics Endpoints ---
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock analytics data
    analytics = {
        'totalPnL': 2450.00,
        'winRate': 0.68,
        'avgWin': 145.00,
        'avgLoss': -85.00,
        'sharpeRatio': 1.85,
        'maxDrawdown': -8.5,
        'totalTrades': 150,
        'monthlyReturns': [
            {'month': 'Jan', 'return': 5.2},
            {'month': 'Feb', 'return': -2.1},
            {'month': 'Mar', 'return': 8.7},
            {'month': 'Apr', 'return': 3.4},
            {'month': 'May', 'return': 6.8},
            {'month': 'Jun', 'return': 4.3}
        ]
    }
    
    return jsonify(analytics)

# --- ZK Proof Endpoints ---
@app.route('/api/zk_proofs', methods=['GET'])
def get_zk_proofs():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('SELECT commitment, created_at FROM zk_commitments WHERE username = ? ORDER BY created_at DESC', (username,))
    
    proofs = []
    for row in c.fetchall():
        proofs.append({
            'commitment': row[0],
            'createdAt': row[1],
            'status': 'verified'
        })
    
    conn.close()
    return jsonify(proofs)

@app.route('/api/subscription_history', methods=['GET'])
def get_subscription_history():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock subscription history
    history = [
        {'date': '2024-01-01', 'amount': 20.00, 'status': 'paid', 'description': 'Monthly subscription'},
        {'date': '2024-02-01', 'amount': 20.00, 'status': 'paid', 'description': 'Monthly subscription'},
        {'date': '2024-03-01', 'amount': 20.00, 'status': 'paid', 'description': 'Monthly subscription'}
    ]
    
    return jsonify(history)

# --- User Auth ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    
    # Hash password for security
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                 (data['username'], password_hash))
        conn.commit()
        
        # Add sample trades for demo
        sample_trades = [
            ('2024-01-01', 2.5, 'Profit'),
            ('2024-01-02', -1.2, 'Loss'),
            ('2024-01-03', 3.8, 'Profit'),
            ('2024-01-04', 0.0, 'BE'),
            ('2024-01-05', 5.2, 'Profit'),
        ]
        for date, pnl, outcome in sample_trades:
            c.execute('INSERT INTO trades (username, timestamp, pnl_percent, outcome) VALUES (?, ?, ?, ?)',
                     (data['username'], date, pnl, outcome))
        conn.commit()
        
    except sqlite3.IntegrityError:
        return jsonify({"error": "User exists"}), 409
    finally:
        conn.close()
    
    token = jwt.encode({
        "user": data['username'], 
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({"jwt": token})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', 
             (data['username'], password_hash))
    
    if c.fetchone():
        token = jwt.encode({
            "user": data['username'], 
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        conn.close()
        return jsonify({"jwt": token})
    
    conn.close()
    return jsonify({"error": "Invalid credentials"}), 401

# --- User Profile ---
@app.route('/api/user_profile', methods=['GET'])
def user_profile():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('SELECT username, subscription_status, created_at FROM users WHERE username=?', (username,))
    user_data = c.fetchone()
    conn.close()
    
    if user_data:
        return jsonify({
            "username": user_data[0],
            "subscription_status": user_data[1],
            "created_at": user_data[2]
        })
    
    return jsonify({"error": "User not found"}), 404

# --- Stripe Integration ---
@app.route('/api/create-checkout-session', methods=['POST'])
def create_checkout_session():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'Talos Capital Pro',
                        'description': 'Advanced ZK proof verification, unlimited trade tracking, premium analytics',
                    },
                    'unit_amount': 2000,  # $20.00 in cents
                    'recurring': {
                        'interval': 'month',
                    },
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:3000/success',
            cancel_url='http://localhost:3000/cancel',
            metadata={
                'username': username
            }
        )
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_ENDPOINT_SECRET
        )
    except ValueError:
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({'error': 'Invalid signature'}), 400
    
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        username = session['metadata']['username']
        
        # Update user subscription status
        conn = sqlite3.connect('trading_journal.db')
        c = conn.cursor()
        c.execute('UPDATE users SET subscription_status = ? WHERE username = ?', 
                 ('paid', username))
        conn.commit()
        conn.close()
    
    return jsonify({'status': 'success'})

# --- Zero-Knowledge Proof Integration ---
@app.route('/api/zk_commitment', methods=['POST'])
def zk_commitment():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    commitment = data.get('commitment', '')
    
    if not commitment:
        return jsonify({"error": "Commitment required"}), 400
    
    # Store ZK commitment
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('INSERT INTO zk_commitments (username, commitment) VALUES (?, ?)', 
             (username, commitment))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "ZK commitment stored successfully", "commitment_hash": commitment})

# --- PnL Chart API ---
@app.route('/api/pnl_data', methods=['GET'])
def pnl_data():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, pnl_percent, outcome FROM trades WHERE username=? AND status='Closed'", (username,))
    trades = c.fetchall()
    conn.close()
    
    timeline = []
    breakdown = {"Win": 0, "Loss": 0, "BE": 0}
    
    for row in trades:
        ts, pnl, outcome = row
        date = ts.split(' ')[0]
        timeline.append({"date": date, "pnl": round(pnl, 2)})
        
        if outcome == 'Profit': 
            breakdown['Win'] += 1
        elif outcome == 'Loss': 
            breakdown['Loss'] += 1
        else: 
            breakdown['BE'] += 1

    return jsonify({"timeline": timeline[-30:], "breakdown": breakdown})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
