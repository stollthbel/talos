# Talos Capital - Phase 4/6 🏛️

**Quantified truth. ZK-powered. Built for Veritas.**

A sophisticated trading journal with Zero-Knowledge proof integration, multi-user authentication, and Stripe-powered subscriptions.

## ✨ Features

### 🔐 Authentication & User Management
- JWT-based secure authentication
- Password hashing with SHA-256
- User profiles with subscription tracking
- Session management with 30-day expiration

### 💳 Stripe Integration
- $20/month Pro subscription
- Secure checkout sessions
- Webhook handling for status updates
- Automatic subscription management

### 🧪 Zero-Knowledge Proof Support
- Cryptographic commitment storage
- Privacy-preserving trade verification
- ZK-SNARK/zk-STARK compatibility layer
- Pro-tier exclusive features

### 📊 Advanced Analytics
- PnL timeline visualization
- Win/Loss breakdown charts
- User-scoped trade data
- Responsive data visualization

### 🎨 Modern UI/UX
- Dark theme with gradient accents
- Responsive design
- Glass morphism effects
- Tailwind CSS styling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Stripe account (for payments)

### Installation

1. **Clone & Setup**
   ```bash
   git clone <your-repo>
   cd talos
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Stripe keys
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   ./start.sh
   ```

5. **Access the App**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## 📋 API Endpoints

### Authentication
- `POST /api/register` - Create new user account
- `POST /api/login` - User login
- `GET /api/user_profile` - Get user profile (requires auth)

### Subscription Management
- `POST /api/create-checkout-session` - Create Stripe checkout
- `POST /api/webhook` - Stripe webhook handler

### Zero-Knowledge Proofs
- `POST /api/zk_commitment` - Submit ZK commitment (Pro only)

### Analytics
- `GET /api/pnl_data` - Get PnL timeline and breakdown

## 🔧 Configuration

### Environment Variables
```env
# Required
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_ENDPOINT_SECRET=whsec_...

# Optional
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///trading_journal.db
CORS_ORIGINS=http://localhost:3000
```

### Database Schema
```sql
users (username, password, subscription_status, created_at)
trades (username, timestamp, pnl_percent, outcome, status)
zk_commitments (username, commitment, created_at)
```

## 🔐 Security Features

- **Password Security**: SHA-256 hashing
- **JWT Authentication**: Secure token-based auth
- **CORS Protection**: Configurable origins
- **Input Validation**: SQL injection prevention
- **Environment Variables**: Secure configuration

## 🧪 Zero-Knowledge Integration

The ZK proof system supports:
- **Commitment Storage**: Cryptographic commitments
- **Privacy Preservation**: Trade data verification without exposure
- **Future Integration**: Ready for snarkjs/zkSync
- **Pro-Tier Exclusive**: Advanced verification features

## 📊 Subscription Tiers

### Free Tier
- Basic trade tracking
- Limited analytics
- Standard authentication

### Pro Tier ($20/month)
- Advanced ZK proof verification
- Unlimited trade tracking
- Premium analytics dashboard
- Priority support

## 🚀 Deployment

### Local Development
```bash
./start.sh
```

### Production Ready
- Configure environment variables
- Set up reverse proxy (nginx)
- Use production database
- Enable HTTPS
- Configure Stripe webhooks

## 📱 Frontend Technology

- **React 18**: Modern hooks-based components
- **Recharts**: Data visualization
- **Axios**: HTTP client
- **Tailwind CSS**: Utility-first styling
- **Responsive Design**: Mobile-first approach

## 🔧 Backend Technology

- **Flask**: Python web framework
- **SQLite**: Lightweight database
- **JWT**: Token-based authentication
- **Stripe**: Payment processing
- **CORS**: Cross-origin support

## 🛠️ Development

### Adding New Features
1. Update database schema if needed
2. Add API endpoints in `talos-backend.py`
3. Update frontend components in `talos-frontend.jsx`
4. Test authentication flows
5. Update documentation

### Testing
```bash
# Backend tests
python -m pytest tests/

# Frontend tests
npm test
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🎯 Roadmap

### Phase 5/6: Advanced Features
- Real ZK-SNARK integration
- Advanced analytics dashboard
- Multi-asset support
- API rate limiting

### Phase 6/6: Production Ready
- Containerization (Docker)
- CI/CD pipeline
- Performance optimization
- Security audit

---

**Built with 🏛️ by the Talos Capital team**