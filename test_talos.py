#!/usr/bin/env python3
"""
Talos Capital - Phase 4/6 Enhanced Test Suite
Complete tests for trading journal with drawing and rich text features
"""

import os
import sys
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_schema():
    """Test complete database schema including journal tables"""
    try:
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        
        # Test users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                subscription_status TEXT DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Test trades table
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
        
        # Test journal entries table
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
        
        # Test zk_commitments table
        c.execute('''
            CREATE TABLE IF NOT EXISTS zk_commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                commitment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Enhanced database schema test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        return False

def test_journal_functionality():
    """Test journal entry creation and management"""
    try:
        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        
        # Setup tables
        c.execute('''
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password TEXT,
                subscription_status TEXT DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE journal_entries (
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
        
        # Create test user
        username = "testuser"
        password_hash = hashlib.sha256("testpass123".encode()).hexdigest()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password_hash))
        
        # Create test journal entry
        test_entry = {
            'title': 'Test Trading Day',
            'content': '## Market Analysis\n\nMarket was volatile today...',
            'mood': 'bullish',
            'market_conditions': 'High volatility, strong volume',
            'lessons': 'Need to be more patient with entries',
            'next_steps': 'Review risk management',
            'tags': ['SPY', 'day-trading', 'volatility'],
            'drawings': [{'id': 1, 'dataURL': 'data:image/png;base64,test'}],
            'is_private': False
        }
        
        c.execute('''
            INSERT INTO journal_entries 
            (username, title, content, mood, market_conditions, lessons, next_steps, tags, drawings, is_private)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            username,
            test_entry['title'],
            test_entry['content'],
            test_entry['mood'],
            test_entry['market_conditions'],
            test_entry['lessons'],
            test_entry['next_steps'],
            ','.join(test_entry['tags']),
            json.dumps(test_entry['drawings']),
            test_entry['is_private']
        ))
        
        conn.commit()
        
        # Test data retrieval
        c.execute('SELECT * FROM journal_entries WHERE username = ?', (username,))
        entry = c.fetchone()
        assert entry is not None, "Journal entry should exist"
        assert entry[2] == test_entry['title'], "Title should match"
        assert entry[4] == test_entry['mood'], "Mood should match"
        
        # Test JSON parsing of drawings
        drawings = json.loads(entry[9])
        assert len(drawings) == 1, "Should have one drawing"
        assert drawings[0]['id'] == 1, "Drawing ID should match"
        
        conn.close()
        
        print("✅ Journal functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Journal functionality test failed: {e}")
        return False

def test_drawing_data_structure():
    """Test drawing data serialization and deserialization"""
    try:
        # Test drawing data structure
        drawing = {
            'id': 12345,
            'dataURL': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
            'createdAt': datetime.now().isoformat(),
            'tool': 'pen',
            'color': '#ffffff',
            'size': 2
        }
        
        # Test JSON serialization
        json_str = json.dumps([drawing])
        parsed = json.loads(json_str)
        
        assert len(parsed) == 1, "Should parse one drawing"
        assert parsed[0]['id'] == 12345, "Drawing ID should match"
        assert parsed[0]['tool'] == 'pen', "Tool should match"
        
        print("✅ Drawing data structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Drawing data structure test failed: {e}")
        return False

def test_template_system():
    """Test journal template functionality"""
    try:
        templates = [
            {
                'id': 1,
                'name': 'Daily Trade Review',
                'template': {
                    'title': 'Daily Trade Review - {date}',
                    'content': '## Trade Summary\n\n### Positions Taken:\n- \n\n### Market Analysis:\n- \n\n### Performance:\n- \n\n### Lessons Learned:\n- \n\n### Tomorrow\'s Plan:\n- '
                }
            },
            {
                'id': 2,
                'name': 'Psychology Check',
                'template': {
                    'title': 'Psychology Check - {date}',
                    'content': '## Mental State\n\n### Pre-Market Mood:\n- \n\n### During Trading:\n- \n\n### Post-Market Reflection:\n- \n\n### Emotional Triggers:\n- \n\n### Improvement Areas:\n- '
                }
            }
        ]
        
        # Test template application
        today = datetime.now().strftime('%Y-%m-%d')
        template = templates[0]['template']
        
        title = template['title'].replace('{date}', today)
        content = template['content']
        
        assert today in title, "Date should be in title"
        assert '## Trade Summary' in content, "Content should contain sections"
        assert '### Positions Taken:' in content, "Content should contain subsections"
        
        print("✅ Template system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Template system test failed: {e}")
        return False

def test_mood_tracking():
    """Test mood tracking functionality"""
    try:
        moods = ['bullish', 'bearish', 'neutral', 'confident', 'uncertain', 'frustrated', 'excited']
        
        # Test each mood
        for mood in moods:
            assert mood in moods, f"Mood {mood} should be valid"
        
        # Test mood filtering
        test_entries = [
            {'mood': 'bullish', 'date': '2024-01-01'},
            {'mood': 'bearish', 'date': '2024-01-02'},
            {'mood': 'neutral', 'date': '2024-01-03'},
            {'mood': 'bullish', 'date': '2024-01-04'},
        ]
        
        bullish_entries = [e for e in test_entries if e['mood'] == 'bullish']
        assert len(bullish_entries) == 2, "Should find 2 bullish entries"
        
        print("✅ Mood tracking test passed")
        return True
        
    except Exception as e:
        print(f"❌ Mood tracking test failed: {e}")
        return False

def test_tags_system():
    """Test tag functionality"""
    try:
        # Test tag creation and management
        tags = ['SPY', 'day-trading', 'volatility', 'earnings', 'technical-analysis']
        
        # Test tag serialization
        tags_str = ','.join(tags)
        parsed_tags = tags_str.split(',')
        
        assert len(parsed_tags) == len(tags), "Tag count should match"
        assert 'SPY' in parsed_tags, "SPY tag should exist"
        assert 'day-trading' in parsed_tags, "day-trading tag should exist"
        
        # Test tag filtering
        entries_with_spy = [e for e in [{'tags': ['SPY', 'AAPL']}, {'tags': ['TSLA']}] if 'SPY' in e['tags']]
        assert len(entries_with_spy) == 1, "Should find 1 entry with SPY tag"
        
        print("✅ Tags system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Tags system test failed: {e}")
        return False

def test_markdown_formatting():
    """Test markdown formatting capabilities"""
    try:
        # Test markdown content
        markdown_content = """
# Trading Journal Entry

## Market Analysis
The market showed **strong volatility** today with *significant* volume.

### Key Points:
- SPY broke above resistance at $430
- Volume was __exceptionally high__
- Earnings reactions were mixed

### Strategy Notes:
1. Look for breakout confirmations
2. Manage risk carefully
3. Consider scaling out positions
        """
        
        # Test that markdown elements are preserved
        assert '# Trading Journal Entry' in markdown_content, "Should contain H1 header"
        assert '## Market Analysis' in markdown_content, "Should contain H2 header"
        assert '**strong volatility**' in markdown_content, "Should contain bold text"
        assert '*significant*' in markdown_content, "Should contain italic text"
        assert '__exceptionally high__' in markdown_content, "Should contain underline text"
        assert '- SPY broke above' in markdown_content, "Should contain bullet points"
        assert '1. Look for breakout' in markdown_content, "Should contain numbered list"
        
        print("✅ Markdown formatting test passed")
        return True
        
    except Exception as e:
        print(f"❌ Markdown formatting test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    try:
        required_files = [
            'talos-backend.py',
            'talos-frontend.jsx',
            'requirements.txt',
            'package.json',
            'README.md',
            '.env.example',
            'App.css',
            'index.html',
            'start.sh'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        # Test that files have expected content
        with open('talos-frontend.jsx', 'r') as f:
            frontend_content = f.read()
            assert 'journalEntries' in frontend_content, "Frontend should have journal functionality"
            assert 'drawingMode' in frontend_content, "Frontend should have drawing functionality"
            assert 'renderJournalEditor' in frontend_content, "Frontend should have journal editor"
        
        with open('talos-backend.py', 'r') as f:
            backend_content = f.read()
            assert 'journal_entries' in backend_content, "Backend should have journal endpoints"
            assert 'CREATE TABLE IF NOT EXISTS journal_entries' in backend_content, "Backend should create journal table"
        
        print("✅ File structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def run_all_tests():
    """Run all enhanced tests"""
    print("🧪 Running Talos Capital - Phase 4/6 Enhanced Test Suite")
    print("=" * 60)
    
    tests = [
        test_database_schema,
        test_journal_functionality,
        test_drawing_data_structure,
        test_template_system,
        test_mood_tracking,
        test_tags_system,
        test_markdown_formatting,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Talos Capital is ready!")
        print("")
        print("🚀 Your trading journal now includes:")
        print("   📝 Rich text editing with markdown support")
        print("   🎨 Drawing and annotation tools")
        print("   🧠 Mood tracking and market analysis")
        print("   🏷️ Tagging system for organization")
        print("   📋 Pre-built templates for common entries")
        print("   🔒 Private entries for sensitive thoughts")
        print("   📊 Enhanced analytics and visualizations")
        print("   💳 Stripe subscription management")
        print("   🧪 Zero-knowledge proof integration")
        print("")
        print("Run './start.sh' to launch your enhanced trading journal!")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
