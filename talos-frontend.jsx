// TalosCapitalApp.jsx — Phase 4/6 Enhanced Frontend (React + Stripe + ZK)
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, PieChart, Pie, Cell, Tooltip, Legend, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area, ScatterChart, Scatter, RadialBarChart, RadialBar
} from 'recharts';
import './App.css';

function TalosCapitalApp() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState(localStorage.getItem('jwt') || '');
  const [isAuthenticated, setAuthenticated] = useState(!!token);
  const [status, setStatus] = useState('');
  const [pnlData, setPnlData] = useState([]);
  const [breakdown, setBreakdown] = useState([]);
  const [userProfile, setUserProfile] = useState({});
  const [showUpgrade, setShowUpgrade] = useState(false);
  const [zkCommitment, setZkCommitment] = useState('');
  const [zkProofStatus, setZkProofStatus] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Enhanced state for new features
  const [activeTab, setActiveTab] = useState('dashboard');
  const [trades, setTrades] = useState([]);
  const [newTrade, setNewTrade] = useState({ symbol: '', type: 'buy', amount: '', price: '', notes: '' });
  const [portfolioData, setPortfolioData] = useState([]);
  const [marketData, setMarketData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [settings, setSettings] = useState({ theme: 'dark', notifications: true, riskLevel: 'medium' });
  const [analyticsData, setAnalyticsData] = useState({});
  const [zkProofs, setZkProofs] = useState([]);
  const [subscriptionHistory, setSubscriptionHistory] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [filterBy, setFilterBy] = useState('all');
  const [showModal, setShowModal] = useState(false);
  const [modalContent, setModalContent] = useState('');
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  
  // Enhanced Journal Features
  const [journalEntries, setJournalEntries] = useState([]);
  const [currentEntry, setCurrentEntry] = useState({
    id: null,
    title: '',
    content: '',
    drawings: [],
    tags: [],
    mood: 'neutral',
    marketConditions: '',
    lessons: '',
    nextSteps: '',
    attachments: [],
    createdAt: new Date(),
    updatedAt: new Date(),
    isPrivate: false
  });
  const [showJournalEditor, setShowJournalEditor] = useState(false);
  const [journalFilter, setJournalFilter] = useState('all');
  const [drawingMode, setDrawingMode] = useState(false);
  const [drawingTool, setDrawingTool] = useState('pen');
  const [drawingColor, setDrawingColor] = useState('#ffffff');
  const [drawingSize, setDrawingSize] = useState(2);
  const [canvasRef, setCanvasRef] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [journalTemplates, setJournalTemplates] = useState([
    {
      id: 1,
      name: 'Daily Trade Review',
      template: {
        title: 'Daily Trade Review - {date}',
        content: '## Trade Summary\n\n### Positions Taken:\n- \n\n### Market Analysis:\n- \n\n### Performance:\n- \n\n### Lessons Learned:\n- \n\n### Tomorrow\'s Plan:\n- '
      }
    },
    {
      id: 2,
      name: 'Strategy Analysis',
      template: {
        title: 'Strategy Analysis - {strategy}',
        content: '## Strategy Overview\n\n### Entry Criteria:\n- \n\n### Exit Criteria:\n- \n\n### Risk Management:\n- \n\n### Performance Metrics:\n- \n\n### Improvements:\n- '
      }
    },
    {
      id: 3,
      name: 'Market Observation',
      template: {
        title: 'Market Observation - {date}',
        content: '## Market Conditions\n\n### Overall Trend:\n- \n\n### Key Levels:\n- \n\n### Volume Analysis:\n- \n\n### Sector Performance:\n- \n\n### News Impact:\n- '
      }
    },
    {
      id: 4,
      name: 'Psychology Check',
      template: {
        title: 'Psychology Check - {date}',
        content: '## Mental State\n\n### Pre-Market Mood:\n- \n\n### During Trading:\n- \n\n### Post-Market Reflection:\n- \n\n### Emotional Triggers:\n- \n\n### Improvement Areas:\n- '
      }
    }
  ]);
  const [showDrawingTools, setShowDrawingTools] = useState(false);
  const [savedDrawings, setSavedDrawings] = useState([]);
  const [textFormatting, setTextFormatting] = useState({
    bold: false,
    italic: false,
    underline: false,
    fontSize: 16,
    fontFamily: 'Arial',
    textColor: '#ffffff'
  });

  useEffect(() => {
    if (token) {
      fetchPnl();
      fetchUserProfile();
      fetchTrades();
      fetchPortfolioData();
      fetchMarketData();
      fetchAnalytics();
      fetchZkProofs();
      fetchSubscriptionHistory();
      fetchJournalEntries();
    }
  }, [token]);

  useEffect(() => {
    // Real-time updates simulation
    const interval = setInterval(() => {
      if (token && isAuthenticated) {
        fetchMarketData();
        updateConnectionStatus();
      }
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [token, isAuthenticated]);

  const updateConnectionStatus = () => {
    setConnectionStatus(Math.random() > 0.1 ? 'connected' : 'disconnected');
  };

  const fetchTrades = async () => {
    try {
      const res = await axios.get('/api/trades', { headers: { Authorization: `Bearer ${token}` } });
      setTrades(res.data);
    } catch (error) {
      console.error('Failed to fetch trades:', error);
    }
  };

  const fetchPortfolioData = async () => {
    try {
      const res = await axios.get('/api/portfolio', { headers: { Authorization: `Bearer ${token}` } });
      setPortfolioData(res.data);
    } catch (error) {
      console.error('Failed to fetch portfolio data:', error);
    }
  };

  const fetchMarketData = async () => {
    try {
      const res = await axios.get('/api/market_data', { headers: { Authorization: `Bearer ${token}` } });
      setMarketData(res.data);
    } catch (error) {
      console.error('Failed to fetch market data:', error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await axios.get('/api/analytics', { headers: { Authorization: `Bearer ${token}` } });
      setAnalyticsData(res.data);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    }
  };

  const fetchZkProofs = async () => {
    try {
      const res = await axios.get('/api/zk_proofs', { headers: { Authorization: `Bearer ${token}` } });
      setZkProofs(res.data);
    } catch (error) {
      console.error('Failed to fetch ZK proofs:', error);
    }
  };

  const fetchSubscriptionHistory = async () => {
    try {
      const res = await axios.get('/api/subscription_history', { headers: { Authorization: `Bearer ${token}` } });
      setSubscriptionHistory(res.data);
    } catch (error) {
      console.error('Failed to fetch subscription history:', error);
    }
  };

  const fetchJournalEntries = async () => {
    try {
      const res = await axios.get('/api/journal_entries', { headers: { Authorization: `Bearer ${token}` } });
      setJournalEntries(res.data);
    } catch (error) {
      console.error('Failed to fetch journal entries:', error);
    }
  };

  const saveJournalEntry = async (entry) => {
    try {
      const method = entry.id ? 'put' : 'post';
      const url = entry.id ? `/api/journal_entries/${entry.id}` : '/api/journal_entries';
      const res = await axios[method](url, entry, { headers: { Authorization: `Bearer ${token}` } });
      
      if (entry.id) {
        setJournalEntries(entries => entries.map(e => e.id === entry.id ? res.data : e));
      } else {
        setJournalEntries(entries => [...entries, res.data]);
      }
      
      setCurrentEntry({
        id: null,
        title: '',
        content: '',
        drawings: [],
        tags: [],
        mood: 'neutral',
        marketConditions: '',
        lessons: '',
        nextSteps: '',
        attachments: [],
        createdAt: new Date(),
        updatedAt: new Date(),
        isPrivate: false
      });
      setShowJournalEditor(false);
    } catch (error) {
      console.error('Failed to save journal entry:', error);
    }
  };

  const deleteJournalEntry = async (id) => {
    try {
      await axios.delete(`/api/journal_entries/${id}`, { headers: { Authorization: `Bearer ${token}` } });
      setJournalEntries(entries => entries.filter(e => e.id !== id));
    } catch (error) {
      console.error('Failed to delete journal entry:', error);
    }
  };

  const applyTemplate = (template) => {
    const today = new Date().toISOString().split('T')[0];
    const title = template.title.replace('{date}', today).replace('{strategy}', 'Your Strategy');
    const content = template.content;
    
    setCurrentEntry(prev => ({
      ...prev,
      title,
      content
    }));
    setShowTemplates(false);
  };

  const startDrawing = (e) => {
    if (!canvasRef) return;
    setIsDrawing(true);
    const rect = canvasRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const ctx = canvasRef.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = drawingColor;
    ctx.lineWidth = drawingSize;
    ctx.lineCap = 'round';
  };

  const draw = (e) => {
    if (!isDrawing || !canvasRef) return;
    
    const rect = canvasRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const ctx = canvasRef.getContext('2d');
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    if (!canvasRef) return;
    const ctx = canvasRef.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
  };

  const saveDrawing = () => {
    if (!canvasRef) return;
    const dataURL = canvasRef.toDataURL();
    const newDrawing = {
      id: Date.now(),
      dataURL,
      createdAt: new Date()
    };
    setSavedDrawings(prev => [...prev, newDrawing]);
    setCurrentEntry(prev => ({
      ...prev,
      drawings: [...prev.drawings, newDrawing]
    }));
  };

  const addTag = (tag) => {
    if (tag && !currentEntry.tags.includes(tag)) {
      setCurrentEntry(prev => ({
        ...prev,
        tags: [...prev.tags, tag]
      }));
    }
  };

  const removeTag = (tagToRemove) => {
    setCurrentEntry(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const formatText = (format) => {
    const textarea = document.getElementById('journal-content');
    if (!textarea) return;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;
    const selectedText = text.substring(start, end);
    
    let formattedText = selectedText;
    
    switch (format) {
      case 'bold':
        formattedText = `**${selectedText}**`;
        break;
      case 'italic':
        formattedText = `*${selectedText}*`;
        break;
      case 'underline':
        formattedText = `__${selectedText}__`;
        break;
      case 'heading':
        formattedText = `## ${selectedText}`;
        break;
      case 'bullet':
        formattedText = `- ${selectedText}`;
        break;
      case 'number':
        formattedText = `1. ${selectedText}`;
        break;
      default:
        break;
    }
    
    const newText = text.substring(0, start) + formattedText + text.substring(end);
    setCurrentEntry(prev => ({
      ...prev,
      content: newText
    }));
  };

  const fetchUserProfile = async () => {
    try {
      const res = await axios.get('/api/user_profile', { headers: { Authorization: `Bearer ${token}` } });
      setUserProfile(res.data);
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
    }
  };

  const fetchPnl = async () => {
    try {
      const res = await axios.get('/api/pnl_data', { headers: { Authorization: `Bearer ${token}` } });
      setPnlData(res.data.timeline);
      setBreakdown([
        { name: 'Win', value: res.data.breakdown.Win },
        { name: 'Loss', value: res.data.breakdown.Loss },
        { name: 'BE', value: res.data.breakdown.BE }
      ]);
    } catch (error) {
      console.error('Failed to fetch PnL data:', error);
    }
  };

  const login = async () => {
    try {
      const res = await axios.post('/api/login', { username, password });
      localStorage.setItem('jwt', res.data.jwt);
      setToken(res.data.jwt);
      setAuthenticated(true);
      setStatus('✅ Login successful');
    } catch {
      setStatus('❌ Login failed');
    }
  };

  const register = async () => {
    try {
      const res = await axios.post('/api/register', { username, password });
      localStorage.setItem('jwt', res.data.jwt);
      setToken(res.data.jwt);
      setAuthenticated(true);
      setStatus('✅ Registration successful');
    } catch {
      setStatus('❌ Register failed');
    }
  };

  const logout = () => {
    setAuthenticated(false);
    setToken('');
    localStorage.removeItem('jwt');
    setUsername('');
    setPassword('');
    setPnlData([]);
    setBreakdown([]);
    setUserProfile({});
  };

  const startCheckout = async () => {
    setLoading(true);
    try {
      const res = await axios.post('/api/create-checkout-session', {
        username: userProfile.username || username
      }, { headers: { Authorization: `Bearer ${token}` } });
      window.location.href = res.data.url;
    } catch (error) {
      console.error('Checkout failed:', error);
      setStatus('❌ Payment initiation failed');
    } finally {
      setLoading(false);
    }
  };

  const submitZkCommitment = async () => {
    if (!zkCommitment.trim()) {
      setZkProofStatus('❌ Please enter a commitment');
      return;
    }
    
    setLoading(true);
    try {
      const res = await axios.post('/api/zk_commitment', {
        commitment: zkCommitment
      }, { headers: { Authorization: `Bearer ${token}` } });
      setZkProofStatus('✅ ZK commitment stored successfully');
      setZkCommitment('');
    } catch (error) {
      console.error('ZK commitment failed:', error);
      setZkProofStatus('❌ ZK commitment failed');
    } finally {
      setLoading(false);
    }
  };

  const isPaidUser = userProfile.subscription_status === 'paid';

  const renderJournalEditor = () => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[#1a1a1a] rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Trading Journal Entry</h2>
          <div className="flex gap-2">
            <button onClick={() => setShowTemplates(true)} className="bg-blue-600 px-3 py-1 rounded text-sm">
              Templates
            </button>
            <button onClick={() => setShowJournalEditor(false)} className="bg-gray-600 px-3 py-1 rounded text-sm">
              Cancel
            </button>
          </div>
        </div>

        {showTemplates && (
          <div className="bg-white/10 p-4 rounded-lg mb-4">
            <h3 className="font-bold mb-2">Choose a Template</h3>
            <div className="grid grid-cols-2 gap-2">
              {journalTemplates.map(template => (
                <button
                  key={template.id}
                  onClick={() => applyTemplate(template.template)}
                  className="bg-purple-600 p-2 rounded text-sm hover:bg-purple-700"
                >
                  {template.name}
                </button>
              ))}
            </div>
            <button onClick={() => setShowTemplates(false)} className="mt-2 text-gray-400 text-sm">
              Close Templates
            </button>
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Title</label>
            <input
              type="text"
              value={currentEntry.title}
              onChange={(e) => setCurrentEntry(prev => ({ ...prev, title: e.target.value }))}
              className="w-full p-2 rounded bg-white/10 border border-white/20"
              placeholder="Enter journal entry title..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Mood</label>
            <select
              value={currentEntry.mood}
              onChange={(e) => setCurrentEntry(prev => ({ ...prev, mood: e.target.value }))}
              className="p-2 rounded bg-white/10 border border-white/20"
            >
              <option value="bullish">🐂 Bullish</option>
              <option value="bearish">🐻 Bearish</option>
              <option value="neutral">😐 Neutral</option>
              <option value="confident">😎 Confident</option>
              <option value="uncertain">🤔 Uncertain</option>
              <option value="frustrated">😤 Frustrated</option>
              <option value="excited">🚀 Excited</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Content
              <div className="float-right">
                <div className="flex gap-1 text-xs">
                  <button onClick={() => formatText('bold')} className="bg-white/20 px-2 py-1 rounded">**B**</button>
                  <button onClick={() => formatText('italic')} className="bg-white/20 px-2 py-1 rounded">*I*</button>
                  <button onClick={() => formatText('underline')} className="bg-white/20 px-2 py-1 rounded">__U__</button>
                  <button onClick={() => formatText('heading')} className="bg-white/20 px-2 py-1 rounded">H1</button>
                  <button onClick={() => formatText('bullet')} className="bg-white/20 px-2 py-1 rounded">• List</button>
                  <button onClick={() => formatText('number')} className="bg-white/20 px-2 py-1 rounded">1. List</button>
                </div>
              </div>
            </label>
            <textarea
              id="journal-content"
              value={currentEntry.content}
              onChange={(e) => setCurrentEntry(prev => ({ ...prev, content: e.target.value }))}
              className="w-full p-3 rounded bg-white/10 border border-white/20 min-h-[300px] font-mono"
              placeholder="Write your thoughts, analysis, lessons learned..."
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Market Conditions</label>
              <textarea
                value={currentEntry.marketConditions}
                onChange={(e) => setCurrentEntry(prev => ({ ...prev, marketConditions: e.target.value }))}
                className="w-full p-2 rounded bg-white/10 border border-white/20 h-20"
                placeholder="Overall market sentiment, key levels, volume..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Lessons Learned</label>
              <textarea
                value={currentEntry.lessons}
                onChange={(e) => setCurrentEntry(prev => ({ ...prev, lessons: e.target.value }))}
                className="w-full p-2 rounded bg-white/10 border border-white/20 h-20"
                placeholder="What did you learn from today's trading..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Next Steps</label>
              <textarea
                value={currentEntry.nextSteps}
                onChange={(e) => setCurrentEntry(prev => ({ ...prev, nextSteps: e.target.value }))}
                className="w-full p-2 rounded bg-white/10 border border-white/20 h-20"
                placeholder="Tomorrow's plan, areas to improve..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Tags</label>
              <div className="flex flex-wrap gap-2 mb-2">
                {currentEntry.tags.map(tag => (
                  <span key={tag} className="bg-purple-600 px-2 py-1 rounded text-xs flex items-center gap-1">
                    {tag}
                    <button onClick={() => removeTag(tag)} className="text-red-300 hover:text-red-100">×</button>
                  </span>
                ))}
              </div>
              <input
                type="text"
                placeholder="Add tags (press Enter)"
                className="w-full p-2 rounded bg-white/10 border border-white/20"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    addTag(e.target.value);
                    e.target.value = '';
                  }
                }}
              />
            </div>
          </div>

          <div className="border-t border-white/20 pt-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-bold">Drawing Board</h3>
              <div className="flex gap-2">
                <button onClick={() => setShowDrawingTools(!showDrawingTools)} className="bg-blue-600 px-3 py-1 rounded text-sm">
                  {showDrawingTools ? 'Hide' : 'Show'} Drawing Tools
                </button>
              </div>
            </div>

            {showDrawingTools && (
              <div className="bg-white/10 p-4 rounded-lg mb-4">
                <div className="flex gap-4 mb-4">
                  <div>
                    <label className="block text-xs mb-1">Tool</label>
                    <select
                      value={drawingTool}
                      onChange={(e) => setDrawingTool(e.target.value)}
                      className="bg-white/20 p-1 rounded text-sm"
                    >
                      <option value="pen">Pen</option>
                      <option value="highlighter">Highlighter</option>
                      <option value="eraser">Eraser</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1">Color</label>
                    <input
                      type="color"
                      value={drawingColor}
                      onChange={(e) => setDrawingColor(e.target.value)}
                      className="w-8 h-8 rounded"
                    />
                  </div>
                  <div>
                    <label className="block text-xs mb-1">Size</label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      value={drawingSize}
                      onChange={(e) => setDrawingSize(e.target.value)}
                      className="w-16"
                    />
                  </div>
                  <div className="flex gap-2 items-end">
                    <button onClick={clearCanvas} className="bg-red-600 px-2 py-1 rounded text-xs">
                      Clear
                    </button>
                    <button onClick={saveDrawing} className="bg-green-600 px-2 py-1 rounded text-xs">
                      Save Drawing
                    </button>
                  </div>
                </div>

                <canvas
                  ref={setCanvasRef}
                  width={600}
                  height={300}
                  className="border border-white/20 rounded bg-white/5 cursor-crosshair"
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                />
              </div>
            )}

            {currentEntry.drawings.length > 0 && (
              <div className="mt-4">
                <h4 className="font-bold mb-2">Saved Drawings</h4>
                <div className="grid grid-cols-2 gap-2">
                  {currentEntry.drawings.map(drawing => (
                    <img
                      key={drawing.id}
                      src={drawing.dataURL}
                      alt="Drawing"
                      className="border border-white/20 rounded max-w-full h-20 object-cover"
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={currentEntry.isPrivate}
                onChange={(e) => setCurrentEntry(prev => ({ ...prev, isPrivate: e.target.checked }))}
                className="rounded"
              />
              <label className="text-sm">Private Entry</label>
            </div>

            <div className="flex gap-2">
              <button onClick={() => setShowJournalEditor(false)} className="bg-gray-600 px-4 py-2 rounded">
                Cancel
              </button>
              <button
                onClick={() => saveJournalEntry(currentEntry)}
                className="bg-green-600 px-4 py-2 rounded"
              >
                Save Entry
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderJournalList = () => (
    <div className="bg-white/10 p-6 rounded-xl">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">📖 Trading Journal</h2>
        <div className="flex gap-2">
          <select
            value={journalFilter}
            onChange={(e) => setJournalFilter(e.target.value)}
            className="bg-white/20 p-2 rounded"
          >
            <option value="all">All Entries</option>
            <option value="bullish">Bullish</option>
            <option value="bearish">Bearish</option>
            <option value="neutral">Neutral</option>
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
          </select>
          <button
            onClick={() => {
              setCurrentEntry({
                id: null,
                title: '',
                content: '',
                drawings: [],
                tags: [],
                mood: 'neutral',
                marketConditions: '',
                lessons: '',
                nextSteps: '',
                attachments: [],
                createdAt: new Date(),
                updatedAt: new Date(),
                isPrivate: false
              });
              setShowJournalEditor(true);
            }}
            className="bg-purple-600 px-4 py-2 rounded flex items-center gap-2"
          >
            <span>+</span> New Entry
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {journalEntries.filter(entry => {
          if (journalFilter === 'all') return true;
          if (journalFilter === 'today') {
            const today = new Date().toDateString();
            return new Date(entry.createdAt).toDateString() === today;
          }
          if (journalFilter === 'week') {
            const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
            return new Date(entry.createdAt) > weekAgo;
          }
          if (journalFilter === 'month') {
            const monthAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
            return new Date(entry.createdAt) > monthAgo;
          }
          return entry.mood === journalFilter;
        }).map(entry => (
          <div key={entry.id} className="bg-white/5 p-4 rounded-lg border border-white/10">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="font-bold text-lg">{entry.title}</h3>
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <span>{new Date(entry.createdAt).toLocaleDateString()}</span>
                  <span>•</span>
                  <span>{entry.mood}</span>
                  {entry.isPrivate && <span className="text-yellow-400">🔒 Private</span>}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setCurrentEntry(entry);
                    setShowJournalEditor(true);
                  }}
                  className="text-blue-400 hover:text-blue-300"
                >
                  Edit
                </button>
                <button
                  onClick={() => deleteJournalEntry(entry.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
            
            <div className="text-sm text-gray-300 mb-2 line-clamp-3">
              {entry.content.substring(0, 200)}...
            </div>
            
            {entry.tags.length > 0 && (
              <div className="flex gap-1 mb-2">
                {entry.tags.map(tag => (
                  <span key={tag} className="bg-purple-600/30 px-2 py-1 rounded text-xs">
                    {tag}
                  </span>
                ))}
              </div>
            )}
            
            {entry.drawings.length > 0 && (
              <div className="text-xs text-gray-400">
                📎 {entry.drawings.length} drawing(s) attached
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-white">
      <header className="bg-[#1a1a1a] border-b border-white/10 p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold tracking-tight">Talos Capital</h1>
            <p className="text-gray-400 hidden md:block">Quantified truth. ZK-powered. Built for Veritas.</p>
          </div>
          
          {isAuthenticated && (
            <div className="flex items-center gap-4">
              <div className="flex gap-2">
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`px-4 py-2 rounded ${activeTab === 'dashboard' ? 'bg-purple-600' : 'bg-white/10'}`}
                >
                  Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('journal')}
                  className={`px-4 py-2 rounded ${activeTab === 'journal' ? 'bg-purple-600' : 'bg-white/10'}`}
                >
                  Journal
                </button>
                <button
                  onClick={() => setActiveTab('trades')}
                  className={`px-4 py-2 rounded ${activeTab === 'trades' ? 'bg-purple-600' : 'bg-white/10'}`}
                >
                  Trades
                </button>
                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`px-4 py-2 rounded ${activeTab === 'analytics' ? 'bg-purple-600' : 'bg-white/10'}`}
                >
                  Analytics
                </button>
              </div>
              
              <div className="flex items-center gap-2">
                <span className="text-sm">Welcome, {userProfile.username || username}</span>
                {isPaidUser ? (
                  <span className="bg-green-600 text-xs px-2 py-1 rounded">PRO</span>
                ) : (
                  <button onClick={() => setShowUpgrade(!showUpgrade)} className="bg-purple-600 text-xs px-2 py-1 rounded">
                    Upgrade to PRO
                  </button>
                )}
                <button onClick={logout} className="text-gray-400 hover:text-white">
                  <span className="sr-only">Logout</span>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                </button>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="p-6">
        {!isAuthenticated ? (
          <div className="max-w-md mx-auto mt-20">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold mb-2">Welcome to Talos Capital</h2>
              <p className="text-gray-400">Your personal trading journal and analytics platform</p>
            </div>
            
            <div className="bg-white/10 p-6 rounded-xl backdrop-blur-sm">
              <div className="space-y-4">
                <input
                  placeholder="Username"
                  className="w-full p-3 rounded bg-white/10 border border-white/20 text-white placeholder-gray-400"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                />
                <input
                  placeholder="Password"
                  type="password"
                  className="w-full p-3 rounded bg-white/10 border border-white/20 text-white placeholder-gray-400"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
                <div className="flex gap-2">
                  <button onClick={login} className="bg-purple-600 hover:bg-purple-700 flex-1 p-3 rounded font-medium">
                    Login
                  </button>
                  <button onClick={register} className="bg-white/20 hover:bg-white/30 flex-1 p-3 rounded font-medium">
                    Register
                  </button>
                </div>
                {status && (
                  <p className="text-sm text-center mt-2 text-red-400">{status}</p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <>
            {showUpgrade && !isPaidUser && (
              <div className="bg-gradient-to-r from-purple-600/20 to-blue-600/20 p-6 rounded-xl mb-6 border border-purple-500/30">
                <h3 className="text-xl font-bold mb-2">🚀 Upgrade to Talos Pro</h3>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h4 className="font-semibold mb-2">Pro Features:</h4>
                    <ul className="text-sm text-gray-300 space-y-1">
                      <li>• Advanced ZK proof verification</li>
                      <li>• Unlimited journal entries with drawings</li>
                      <li>• Premium analytics dashboard</li>
                      <li>• Export your trading data</li>
                      <li>• Priority support</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Journal Features:</h4>
                    <ul className="text-sm text-gray-300 space-y-1">
                      <li>• Rich text editor with formatting</li>
                      <li>• Drawing and annotation tools</li>
                      <li>• Custom templates</li>
                      <li>• Mood tracking</li>
                      <li>• Private entries</li>
                    </ul>
                  </div>
                </div>
                <div className="flex gap-4">
                  <button 
                    onClick={startCheckout} 
                    disabled={loading}
                    className="bg-purple-600 hover:bg-purple-700 px-6 py-3 rounded font-medium disabled:opacity-50"
                  >
                    {loading ? 'Processing...' : 'Subscribe for $20/month'}
                  </button>
                  <button 
                    onClick={() => setShowUpgrade(false)}
                    className="bg-white/20 hover:bg-white/30 px-6 py-3 rounded font-medium"
                  >
                    Maybe Later
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'dashboard' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white/10 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-400">Total P&L</h3>
                    <p className="text-2xl font-bold text-green-400">+$2,450</p>
                  </div>
                  <div className="bg-white/10 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-400">Win Rate</h3>
                    <p className="text-2xl font-bold text-blue-400">68%</p>
                  </div>
                  <div className="bg-white/10 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-400">Journal Entries</h3>
                    <p className="text-2xl font-bold text-purple-400">{journalEntries.length}</p>
                  </div>
                  <div className="bg-white/10 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-400">Connection</h3>
                    <p className={`text-2xl font-bold ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                      {connectionStatus === 'connected' ? '🟢 Live' : '🔴 Offline'}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white/10 p-6 rounded-xl">
                    <h2 className="text-xl mb-4 font-bold">P&L Over Time</h2>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={pnlData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                        <XAxis dataKey="date" stroke="#ffffff80" />
                        <YAxis stroke="#ffffff80" />
                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #ffffff20' }} />
                        <Line type="monotone" dataKey="pnl" stroke="#8884d8" strokeWidth={3} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="bg-white/10 p-6 rounded-xl">
                    <h2 className="text-xl mb-4 font-bold">Win/Loss Breakdown</h2>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie data={breakdown} dataKey="value" nameKey="name" outerRadius={100} fill="#8884d8" label>
                          <Cell fill="#4ade80" />
                          <Cell fill="#f87171" />
                          <Cell fill="#eab308" />
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {isPaidUser && (
                  <div className="bg-blue-600/20 p-6 rounded-xl border border-blue-500/30">
                    <h3 className="text-lg font-bold mb-4">🧪 Zero-Knowledge Proof Commitment</h3>
                    <p className="text-sm text-gray-300 mb-4">
                      Submit a cryptographic commitment to prove your trading strategy without revealing details.
                    </p>
                    <div className="flex gap-2">
                      <input
                        placeholder="Enter ZK commitment hash..."
                        className="flex-1 p-3 rounded bg-white/10 border border-white/20 text-white placeholder-gray-400"
                        value={zkCommitment}
                        onChange={e => setZkCommitment(e.target.value)}
                      />
                      <button
                        onClick={submitZkCommitment}
                        disabled={loading}
                        className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded font-medium disabled:opacity-50"
                      >
                        {loading ? 'Processing...' : 'Submit'}
                      </button>
                    </div>
                    {zkProofStatus && (
                      <p className="text-sm mt-2 text-green-400">{zkProofStatus}</p>
                    )}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'journal' && (
              <div>
                {renderJournalList()}
                {showJournalEditor && renderJournalEditor()}
              </div>
            )}

            {activeTab === 'trades' && (
              <div className="bg-white/10 p-6 rounded-xl">
                <h2 className="text-2xl font-bold mb-6">Trade History</h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/20">
                        <th className="text-left p-3">Date</th>
                        <th className="text-left p-3">Symbol</th>
                        <th className="text-left p-3">Type</th>
                        <th className="text-left p-3">P&L</th>
                        <th className="text-left p-3">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trades.map(trade => (
                        <tr key={trade.id} className="border-b border-white/10">
                          <td className="p-3">{trade.date}</td>
                          <td className="p-3">{trade.symbol}</td>
                          <td className="p-3">{trade.type}</td>
                          <td className={`p-3 ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {trade.pnl >= 0 ? '+' : ''}${trade.pnl}
                          </td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs ${
                              trade.status === 'Closed' ? 'bg-green-600' : 'bg-yellow-600'
                            }`}>
                              {trade.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'analytics' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white/10 p-6 rounded-xl">
                    <h3 className="text-lg font-bold mb-4">Monthly Performance</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={pnlData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                        <XAxis dataKey="date" stroke="#ffffff80" />
                        <YAxis stroke="#ffffff80" />
                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #ffffff20' }} />
                        <Bar dataKey="pnl" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="bg-white/10 p-6 rounded-xl">
                    <h3 className="text-lg font-bold mb-4">Risk Analysis</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span>Max Drawdown</span>
                        <span className="text-red-400">-8.5%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Sharpe Ratio</span>
                        <span className="text-green-400">1.85</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Average Win</span>
                        <span className="text-green-400">+$145</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Average Loss</span>
                        <span className="text-red-400">-$85</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default TalosCapitalApp;


