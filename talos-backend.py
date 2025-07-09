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
    c.execute('''
        CREATE TABLE IF NOT EXISTS rl_training_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            symbol TEXT,
            days INTEGER,
            timesteps INTEGER,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
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
        return jsondiff({'error': 'Invalid signature'}), 400
    
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
    
    return jsondifff({'status': 'success'})

# --- Zero-Knowledge Proof Integration ---
@app.route('/api/zk_commitment', methods=['POST'])
def zk_commitment():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsondifffffffffff({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    commitment = data.get('commitment', '')
    
    if not commitment:
        return jsondifffff({"error": "Commitment required"}), 400
    
    # Store ZK commitment
    conn = sqlite3.connect('trading_journal.db')
    c = conn.cursor()
    c.execute('INSERT INTO zk_commitments (username, commitment) VALUES (?, ?)', 
             (username, commitment))
    conn.commit()
    conn.close()
    
    return jsondiff({"message": "ZK commitment stored successfully", "commitment_hash": commitment})

# --- PnL Chart API ---
@app.route('/api/pnl_data', methods=['GET'])
def pnl_data():
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsondiff({"error": "Unauthorized"}), 401

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

    return jsondiff({"timeline": timeline[-30:], "breakdown": breakdown})

# --- RL Agent Endpoints ---
@app.route('/api/rl/train', methods=['POST'])
def train_rl_agent():
    """Train RL agent with historical data"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    symbol = data.get('symbol', 'SPY')
    days = data.get('days', 30)
    timesteps = data.get('timesteps', 10000)
    
    try:
        # This would be replaced with actual RL training logic
        training_status = {
            "status": "started",
            "symbol": symbol,
            "days": days,
            "timesteps": timesteps,
            "estimated_time": "5-10 minutes"
        }
        
        # Store training job in database
        conn = sqlite3.connect('trading_journal.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO rl_training_jobs (username, symbol, days, timesteps, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, symbol, days, timesteps, 'started', datetime.datetime.utcnow()))
        job_id = c.lastrowid
        conn.commit()
        conn.close()
        
        training_status["job_id"] = job_id
        return jsonify(training_status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rl/predict', methods=['POST'])
def rl_predict():
    """Get RL agent prediction for current market state"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    market_state = data.get('market_state', [])
    
    # Mock RL prediction (replace with actual model inference)
    import random
    actions = ['HOLD', 'LONG', 'SHORT', 'CLOSE']
    prediction = {
        "action": random.choice(actions),
        "confidence": round(random.uniform(0.6, 0.95), 2),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_version": "v1.0"
    }
    
    return jsonify(prediction)

@app.route('/api/rl/backtest', methods=['POST'])
def rl_backtest():
    """Run backtest with RL agent"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    symbol = data.get('symbol', 'SPY')
    start_date = data.get('start_date', '2024-01-01')
    end_date = data.get('end_date', '2024-06-01')
    
    # Mock backtest results (replace with actual backtesting)
    backtest_results = {
        "symbol": symbol,
        "period": f"{start_date} to {end_date}",
        "total_return": 12.5,
        "sharpe_ratio": 1.85,
        "max_drawdown": -5.2,
        "win_rate": 0.68,
        "total_trades": 45,
        "avg_trade_duration": "2.5 hours",
        "equity_curve": [
            {"date": "2024-01-01", "equity": 10000},
            {"date": "2024-02-01", "equity": 10250},
            {"date": "2024-03-01", "equity": 10180},
            {"date": "2024-04-01", "equity": 10580},
            {"date": "2024-05-01", "equity": 10920},
            {"date": "2024-06-01", "equity": 11250}
        ]
    }
    
    return jsonify(backtest_results)

@app.route('/api/rl/models', methods=['GET'])
def get_rl_models():
    """Get list of trained RL models"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock model list (replace with actual model management)
    models = [
        {
            "id": 1,
            "name": "SPY_PPO_v1",
            "algorithm": "PPO",
            "symbol": "SPY",
            "training_date": "2024-06-01",
            "performance": {
                "sharpe_ratio": 1.85,
                "max_drawdown": -5.2,
                "win_rate": 0.68
            },
            "status": "active"
        },
        {
            "id": 2,
            "name": "QQQ_A2C_v1",
            "algorithm": "A2C",
            "symbol": "QQQ",
            "training_date": "2024-05-15",
            "performance": {
                "sharpe_ratio": 1.65,
                "max_drawdown": -7.8,
                "win_rate": 0.62
            },
            "status": "inactive"
        }
    ]
    
    return jsonify(models)

@app.route('/api/signals/live', methods=['GET'])
def get_live_signals():
    """Get live trading signals from signal engine"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    # Mock live signals (replace with actual signal engine)
    import random
    signals = []
    
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    
    for symbol in symbols:
        if random.random() > 0.7:  # 30% chance of signal
            signal = {
                "symbol": symbol,
                "direction": random.choice(['LONG', 'SHORT']),
                "price": round(random.uniform(100, 500), 2),
                "take_profit": round(random.uniform(105, 520), 2),
                "stop_loss": round(random.uniform(95, 480), 2),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "indicators": {
                    "rsi": round(random.uniform(20, 80), 1),
                    "ema_fast": round(random.uniform(100, 500), 2),
                    "ema_slow": round(random.uniform(100, 500), 2),
                    "vwap": round(random.uniform(100, 500), 2)
                }
            }
            signals.append(signal)
    
    return jsonify(signals)

@app.route('/api/data/historical', methods=['GET'])
def get_historical_data():
    """Get historical tick data for backtesting"""
    auth = request.headers.get('Authorization', '')
    try:
        token = auth.split(' ')[1] if ' ' in auth else auth
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload['user']
    except:
        return jsonify({"error": "Unauthorized"}), 401

    symbol = request.args.get('symbol', 'SPY')
    days = int(request.args.get('days', 30))
    interval = request.args.get('interval', '1m')
    
    # Mock historical data (replace with actual data fetching)
    historical_data = {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "data": [
            {
                "timestamp": "2024-06-01T09:30:00Z",
                "open": 430.50,
                "high": 431.25,
                "low": 430.10,
                "close": 430.85,
                "volume": 1250000
            },
            {
                "timestamp": "2024-06-01T09:31:00Z",
                "open": 430.85,
                "high": 431.50,
                "low": 430.60,
                "close": 431.20,
                "volume": 1180000
            }
            # ... more data points
        ]
    }
    
    return jsonify(historical_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
