# Stoll AI - Multi-Agent Executive System 🤖

**Production-ready AI orchestration platform for trading, SaaS, and personal operations.**

A comprehensive multi-agent system featuring specialized AI agents (SaaS CTO, CFO, COO, Security, Research) orchestrated by a central executive brain. Built for scalable deployment with Docker, monitoring, and real-time decision making.

## 🎯 System Overview

Stoll AI is a sophisticated multi-agent system designed to manage complex operations across multiple domains:

- **🏢 SaaS Management**: Automated deployment, scaling, and monitoring of SaaS products
- **💰 Financial Operations**: Portfolio management, P&L reporting, and risk assessment  
- **⚙️ Personal Operations**: Schedule management, task automation, and lifestyle optimization
- **🔒 Security**: System monitoring, threat detection, and access control
- **📊 Research**: Market analysis, sentiment tracking, and competitive intelligence

### Architecture

- **Central Executive (CEO)**: Coordinates all agents and makes strategic decisions
- **Specialized Agents**: Domain-specific AI agents with unique capabilities
- **Message Broker**: Asynchronous communication between agents
- **Memory Manager**: Persistent knowledge storage and retrieval
- **REST API**: External control and monitoring interface
- **Production Infrastructure**: Docker, Nginx, Redis, Prometheus, Grafana

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 4GB+ RAM recommended

### One-Command Deployment

```bash
# Clone and deploy
git clone <your-repo>
cd talos
./deploy.sh
```

The system will be available at:
- **Main Interface**: http://localhost:80
- **Stoll AI API**: http://localhost:8001  
- **Talos Backend**: http://localhost:5000
- **Grafana Dashboard**: http://localhost:3000
- **Prometheus**: http://localhost:9091

## 🏗️ Detailed Setup

### 1. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

**Required Environment Variables:**
```bash
# Security
JWT_SECRET=your-production-jwt-secret-here
FLASK_SECRET_KEY=your-production-flask-secret-here

# AI Services (Optional)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Trading APIs (Optional)
ALPACA_API_KEY=your-alpaca-key-here
ALPACA_SECRET_KEY=your-alpaca-secret-here
```

### 2. Production Deployment

```bash
# Deploy all services
./deploy.sh

# Check system health
python health_check.py

# View logs
./deploy.sh logs

# Update deployment
./deploy.sh update
```

### 3. Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
python stoll_launcher.py --mode development

# Run specific components
python stoll_api.py              # REST API only
python stoll_ai.py               # Core system only
```

## 🧠 Agent Capabilities

### SaaS CTO Agent
- **Service Deployment**: Automated Docker deployments
- **Auto-scaling**: Dynamic resource adjustment
- **Health Monitoring**: Real-time service health checks
- **Performance Optimization**: Resource usage analysis

### CFO Agent  
- **P&L Reporting**: Comprehensive financial analysis
- **Portfolio Management**: Risk assessment and optimization
- **Tax Optimization**: Automated tax-loss harvesting
- **Performance Attribution**: Detailed performance breakdown

### COO Agent
- **Schedule Management**: Calendar optimization
- **Task Automation**: Routine task automation
- **Home Management**: Smart home integration
- **Productivity Analytics**: Personal efficiency tracking

### Security Agent
- **Threat Detection**: Real-time security monitoring
- **Access Control**: Authentication and authorization
- **Compliance Monitoring**: Regulatory compliance checks
- **Incident Response**: Automated security responses

### Research Agent
- **Market Analysis**: Real-time market data processing
- **Sentiment Analysis**: Social media and news sentiment
- **Economic Indicators**: Macro-economic data analysis
- **Competitive Intelligence**: Market research and analysis
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