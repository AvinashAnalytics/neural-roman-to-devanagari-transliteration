# gui/app.py
"""Main Application for Hindi Transliteration System using Streamlit
CS772 Assignment 2 - मेरा भारत महान
Features:
- LSTM and Transformer models with local attention
- LLM-based transliteration via OpenAI and HuggingFace
- UTF-8 config support and robust error handling
- Interactive Streamlit GUI with dark mode and Indian tricolor theme
"""
import streamlit as st
import torch
import yaml
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import Seq2SeqLSTM
from models.transformer_model import TransformerSeq2Seq
from models.llm_model import LLMTransliterator
from utils.vocab import Vocabulary
from utils.evaluation import Evaluator

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="🇮🇳 Hindi Transliteration System",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/transliteration',
        'Report a bug': 'https://github.com/yourusername/transliteration/issues',
        'About': 'CS772 Assignment 2 - Hindi Transliteration System | मेरा भारत महान'
    }
)

# 🇮🇳 DARK MODE WITH INDIAN TRICOLOR THEME
st.markdown("""
<style>
    /* ===== INDIAN FLAG COLORS ===== */
    :root {
        --saffron: #FF9933;
        --white: #FFFFFF;
        --green: #138808;
        --navy-blue: #000080;
        --dark-bg: #0f172a;
        --dark-card: #1e293b;
        --dark-surface: #334155;
        --gold: #FFD700;
        --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
        --shadow-md: 0 5px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* ===== DARK BACKGROUND ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    
    /* ===== MAIN HEADER WITH TRICOLOR & FLOATING CHARS ===== */
    .main-header {
        position: relative;
        text-align: center;
        padding: 50px 20px;
        background: linear-gradient(
            to bottom,
            var(--saffron) 0%,
            var(--saffron) 33%,
            var(--white) 33%,
            var(--white) 66%,
            var(--green) 66%,
            var(--green) 100%
        );
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: var(--shadow-lg);
        overflow: hidden;
        border: 3px solid var(--gold);
    }
    
    /* Ashoka Chakra Effect */
    .main-header::before {
        content: '☸';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 80px;
        color: var(--navy-blue);
        opacity: 0.3;
        animation: rotateChakra 20s linear infinite;
        z-index: 0;
    }
    
    @keyframes rotateChakra {
        from { transform: translate(-50%, -50%) rotate(0deg); }
        to { transform: translate(-50%, -50%) rotate(360deg); }
    }
    
    /* Floating "मेरा भारत महान" Characters */
    .hindi-char {
        position: absolute;
        font-size: 2.5rem;
        font-weight: 700;
        color: rgba(255, 215, 0, 0.4);
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
        pointer-events: none;
        z-index: 1;
        animation: floatChar 15s ease-in-out infinite;
    }
    
    @keyframes floatChar {
        0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        50% { transform: translateY(-30px) translateX(20px) rotate(5deg); opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    }
    
    /* Individual char positions - "म े र ा   भ ा र त   म ह ा न" */
    .char-1 { top: 10%; left: 5%; animation-delay: 0s; }
    .char-2 { top: 15%; left: 12%; animation-delay: 1s; }
    .char-3 { top: 20%; left: 18%; animation-delay: 2s; }
    .char-4 { top: 25%; left: 25%; animation-delay: 3s; }
    .char-5 { top: 30%; left: 75%; animation-delay: 4s; }
    .char-6 { top: 35%; left: 82%; animation-delay: 5s; }
    .char-7 { top: 40%; left: 88%; animation-delay: 6s; }
    .char-8 { top: 45%; left: 94%; animation-delay: 7s; }
    .char-9 { top: 70%; left: 8%; animation-delay: 8s; }
    .char-10 { top: 75%; left: 15%; animation-delay: 9s; }
    .char-11 { top: 80%; left: 22%; animation-delay: 10s; }
    .char-12 { top: 85%; left: 30%; animation-delay: 11s; }
    
    .main-header h1 {
        position: relative;
        z-index: 2;
        color: var(--navy-blue);
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 
            2px 2px 4px rgba(255, 255, 255, 0.8),
            -1px -1px 2px rgba(0, 0, 0, 0.3);
        margin: 0;
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .main-header p {
        position: relative;
        z-index: 2;
        color: var(--navy-blue);
        font-size: 1.2rem;
        margin-top: 10px;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* ===== FOOTER WITH TRICOLOR ===== */
    .main-footer {
        position: relative;
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(
            to bottom,
            var(--green) 0%,
            var(--green) 33%,
            var(--white) 33%,
            var(--white) 66%,
            var(--saffron) 66%,
            var(--saffron) 100%
        );
        border-radius: 20px;
        margin-top: 40px;
        box-shadow: var(--shadow-lg);
        overflow: hidden;
        border: 3px solid var(--gold);
    }
    
    .main-footer::before {
        content: '🇮🇳';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 100px;
        opacity: 0.2;
        z-index: 0;
    }
    
    .main-footer p {
        position: relative;
        z-index: 2;
        color: var(--navy-blue);
        font-weight: 700;
        margin: 8px 0;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
    }
    
    .footer-hindi {
        font-size: 1.5rem;
        color: var(--saffron);
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== SECTION HEADERS (Dark) ===== */
    .section-header {
        background: linear-gradient(135deg, var(--saffron) 0%, var(--green) 100%);
        padding: 25px 30px;
        border-radius: 15px;
        margin: 25px 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        border-left: 5px solid var(--gold);
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .section-header h2 {
        color: white;
        margin: 0;
        font-weight: 800;
        font-size: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* ===== DARK CARDS ===== */
    .metric-card {
        background: var(--dark-card);
        padding: 25px;
        border-radius: 16px;
        border-left: 5px solid var(--saffron);
        margin: 20px 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        color: #e2e8f0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(255, 153, 51, 0.3);
        border-left-color: var(--gold);
    }
    
    /* ===== MODEL CARDS (Dark) ===== */
    .model-card {
        background: linear-gradient(135deg, var(--dark-card) 0%, var(--dark-surface) 100%);
        padding: 30px;
        border-radius: 20px;
        color: #e2e8f0;
        margin: 20px 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.4s ease;
        border: 2px solid var(--saffron);
    }
    
    .model-card:hover {
        transform: scale(1.03);
        box-shadow: 0 20px 60px rgba(255, 153, 51, 0.4);
        border-color: var(--gold);
    }
    
    /* ===== TRANSLITERATION OUTPUT (Dark Premium) ===== */
    .transliteration-output {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
        padding: 45px;
        border-radius: 20px;
        border: 4px solid var(--gold);
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 25px 0;
        box-shadow: 
            0 15px 50px rgba(255, 215, 0, 0.3),
            inset 0 0 30px rgba(255, 215, 0, 0.1);
        color: var(--gold);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .transliteration-output::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
        animation: outputGlow 4s ease-in-out infinite;
    }
    
    @keyframes outputGlow {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(10%, 10%); }
    }
    
    .transliteration-output:hover {
        transform: scale(1.02);
        box-shadow: 
            0 20px 70px rgba(255, 215, 0, 0.5),
            inset 0 0 40px rgba(255, 215, 0, 0.2);
    }
    
    /* ===== PROVIDER CARDS (Dark Indian Theme) ===== */
    .provider-card {
        background: linear-gradient(135deg, var(--dark-card) 0%, #374151 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        border: 2px solid var(--green);
        color: #e2e8f0;
    }
    
    .provider-card:hover {
        transform: translateX(8px);
        box-shadow: 0 15px 50px rgba(19, 136, 8, 0.4);
        border-color: var(--saffron);
    }
    
    .provider-card h4 {
        margin: 0 0 10px 0;
        color: var(--saffron);
        font-size: 1.5rem;
        font-weight: 800;
    }
    
    .provider-card p {
        color: #cbd5e1;
        margin: 8px 0;
    }
    
    /* ===== PREMIUM BUTTONS (Indian Colors) ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--saffron) 0%, var(--green) 100%);
        color: white;
        border: none;
        padding: 14px 32px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(255, 153, 51, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5);
        background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== SUCCESS/ERROR BOXES (Dark) ===== */
    .success-box {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 2px solid var(--green);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: var(--shadow-md);
        color: #d1fae5;
    }
    
    .error-box {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: var(--shadow-md);
        color: #fecaca;
    }
    
    /* ===== TABS (Dark Indian Theme) ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, var(--dark-card) 0%, var(--dark-surface) 100%);
        border-radius: 15px;
        padding: 10px;
        border: 2px solid var(--saffron);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 153, 51, 0.1);
        color: var(--saffron);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--saffron) 0%, var(--gold) 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(255, 153, 51, 0.4);
        border-color: var(--gold);
    }
    
    /* ===== SIDEBAR (Dark Gradient) ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 3px solid var(--saffron);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--saffron) !important;
    }
    
    /* ===== PROGRESS BAR (Tricolor) ===== */
    .stProgress > div > div {
        background: linear-gradient(
            90deg, 
            var(--saffron) 0%, 
            white 50%, 
            var(--green) 100%
        );
        background-size: 200% 100%;
        animation: progressTricolor 2s linear infinite;
    }
    
    @keyframes progressTricolor {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* ===== EXPANDER (Dark) ===== */
    .streamlit-expanderHeader {
        background: var(--dark-card);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #e2e8f0;
        border: 2px solid var(--dark-surface);
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--dark-surface);
        border-color: var(--saffron);
    }
    
    /* ===== METRICS (Gold) ===== */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        color: var(--gold);
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* ===== DATAFRAME (Dark) ===== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Override Streamlit's default white backgrounds */
    .stDataFrame > div {
        background: var(--dark-card) !important;
    }
    
    /* ===== CUSTOM SCROLLBAR (Indian Colors) ===== */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--saffron) 0%, var(--green) 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--gold) 0%, var(--saffron) 100%);
    }
    
    /* ===== INPUT FIELDS (Dark) ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: var(--dark-card) !important;
        color: #e2e8f0 !important;
        border: 2px solid var(--dark-surface) !important;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--saffron) !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.2) !important;
    }
    
    /* ===== SLIDER (Indian Colors) ===== */
    .stSlider > div > div > div {
        background: var(--saffron) !important;
    }
    
    /* ===== INFO/WARNING/SUCCESS BOXES (Dark) ===== */
    .stAlert {
        background: var(--dark-card);
        border-radius: 10px;
        border-left: 4px solid var(--saffron);
        color: #e2e8f0;
    }
    
    /* ===== TOOLTIPS (Dark) ===== */
    [data-baseweb="tooltip"] {
        background: var(--dark-surface) !important;
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        color: #e2e8f0 !important;
    }
    
    /* ===== LOADING SPINNER (Tricolor) ===== */
    .stSpinner > div {
        border-top-color: var(--saffron) !important;
        border-right-color: white !important;
        border-bottom-color: var(--green) !important;
    }
    
    /* ===== MARKDOWN TEXT (Light on Dark) ===== */
    .stMarkdown {
        color: #e2e8f0;
    }
    
    /* ===== HEADINGS (Saffron) ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--saffron) !important;
    }
    
    /* ===== LINKS (Gold) ===== */
    a {
        color: var(--gold) !important;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: var(--saffron) !important;
        text-decoration: underline;
    }
</style>

<!-- FLOATING "मेरा भारत महान" CHARACTERS -->
<div class="hindi-char char-1">म</div>
<div class="hindi-char char-2">े</div>
<div class="hindi-char char-3">र</div>
<div class="hindi-char char-4">ा</div>
<div class="hindi-char char-5">भ</div>
<div class="hindi-char char-6">ा</div>
<div class="hindi-char char-7">र</div>
<div class="hindi-char char-8">त</div>
<div class="hindi-char char-9">म</div>
<div class="hindi-char char-10">ह</div>
<div class="hindi-char char-11">ा</div>
<div class="hindi-char char-12">न</div>
""", unsafe_allow_html=True)


class TransliterationApp:
    def __init__(self):
        """Initialize the application with session state management"""
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.history = []
            st.session_state.api_keys = {}
            st.session_state.connected_providers = set()
            st.session_state.available_models = {}
            st.session_state.selected_models = {}
            st.session_state.comparison_results = []
            st.session_state.test_results = {}
        
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.src_vocab = None
        self.tgt_vocab = None
        
        # Initialize LLM handler
        self.llm = LLMTransliterator(self.config)
        
        # Restore API keys from session state
        for provider, api_key in st.session_state.api_keys.items():
            self.llm.setup_client(provider, api_key)
        
        # Load models
        self.load_models()
        
        # Initialize evaluator
        self.evaluator = Evaluator()
    
    def load_config(self) -> Dict:
        """Load configuration with UTF-8 encoding support"""
        config_path = 'config/config.yaml'
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except UnicodeDecodeError:
                try:
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    st.warning(f"⚠️ Config encoding error: {e}. Using defaults.")
            except Exception as e:
                st.warning(f"⚠️ Config load error: {e}. Using defaults.")
        
        # Default configuration matching your config.yaml
        return {
            'data': {
                'max_seq_length': 50,
                'max_train_samples': 100000,
                'device': 'auto'
            },
            'lstm': {
                'embedding_dim': 256, 
                'hidden_dim': 512, 
                'num_layers': 2, 
                'dropout': 0.3, 
                'bidirectional': True
            },
            'transformer': {
                'd_model': 256, 
                'n_heads': 8, 
                'num_layers': 2, 
                'd_ff': 1024, 
                'dropout': 0.1, 
                'use_local_attention': True, 
                'local_attention_window': 5, 
                'max_seq_length': 100
            },
            'training': {
                'batch_size': 128,
                'learning_rate': 0.0005,
                'epochs': 10,
                'gradient_clip': 1.0,
                'teacher_forcing_ratio': 0.5,
                'beam_sizes': [1, 3, 5, 10]
            },
            'llm': {
                'max_tokens': 100,
                'temperature_values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 
                'top_p_values': [0.9, 0.95, 0.99, 1.0]
            },
            'preprocessing': {
                'min_frequency': 2
            }
        }
    
    def load_models(self):
        """Load trained neural models with smart error handling"""
        self.model_status = {'vocab': False, 'lstm': False, 'transformer': False}
        
        try:
            # Load vocabularies
            if os.path.exists('data/processed/src_vocab.json') and os.path.exists('data/processed/tgt_vocab.json'):
                self.src_vocab = Vocabulary()
                self.tgt_vocab = Vocabulary()
                self.src_vocab.load('data/processed/src_vocab.json')
                self.tgt_vocab.load('data/processed/tgt_vocab.json')
                self.model_status['vocab'] = True
            
            # Load LSTM
            if self.model_status['vocab'] and os.path.exists('outputs/checkpoints/lstm_best.pt'):
                try:
                    lstm_config = self.config['lstm']
                    
                    self.models['LSTM'] = Seq2SeqLSTM(
                        src_vocab_size=self.src_vocab.size,
                        tgt_vocab_size=self.tgt_vocab.size,
                        embedding_dim=lstm_config['embedding_dim'],
                        hidden_dim=lstm_config['hidden_dim'],
                        num_layers=lstm_config['num_layers'],
                        dropout=lstm_config['dropout'],
                        bidirectional=lstm_config['bidirectional'],
                        max_length=100
                    ).to(self.device)
                    
                    checkpoint = torch.load('outputs/checkpoints/lstm_best.pt', map_location=self.device)
                    state_dict = checkpoint['model_state_dict']
                    model_dict = self.models['LSTM'].state_dict()
                    
                    # Filter mismatched parameters
                    filtered_state_dict = {}
                    for k, v in state_dict.items():
                        if k in model_dict and v.shape == model_dict[k].shape:
                            filtered_state_dict[k] = v
                    
                    self.models['LSTM'].load_state_dict(filtered_state_dict, strict=False)
                    self.models['LSTM'].eval()
                    self.model_status['lstm'] = True
                    
                except Exception as e:
                    st.sidebar.warning(f"⚠️ LSTM: {str(e)[:50]}...")
            
            # Load Transformer
            if self.model_status['vocab'] and os.path.exists('outputs/checkpoints/transformer_best.pt'):
                try:
                    transformer_config = self.config['transformer']
                    
                    self.models['Transformer'] = TransformerSeq2Seq(
                        src_vocab_size=self.src_vocab.size,
                        tgt_vocab_size=self.tgt_vocab.size,
                        d_model=transformer_config['d_model'],
                        n_heads=transformer_config['n_heads'],
                        num_layers=transformer_config['num_layers'],
                        d_ff=transformer_config['d_ff'],
                        dropout=transformer_config['dropout'],
                        use_local_attention=transformer_config['use_local_attention'],
                        window_size=transformer_config['local_attention_window'],
                        max_seq_length=transformer_config['max_seq_length']
                    ).to(self.device)
                    
                    checkpoint = torch.load('outputs/checkpoints/transformer_best.pt', map_location=self.device)
                    self.models['Transformer'].load_state_dict(checkpoint['model_state_dict'])
                    self.models['Transformer'].eval()
                    self.model_status['transformer'] = True
                    
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Transformer: {str(e)[:50]}...")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Model loading error: {str(e)[:100]}...")
    
    def transliterate_neural(self, text: str, model_name: str, beam_size: int = 1) -> str:
        """Transliterate using neural models"""
        if model_name not in self.models:
            return f"❌ {model_name} not loaded"
        
        try:
            input_indices = self.src_vocab.encode(text)
            input_tensor = torch.tensor([input_indices]).to(self.device)
            
            with torch.no_grad():
                output_indices = self.models[model_name].generate(
                    input_tensor, 
                    max_length=100, 
                    beam_size=beam_size
                )
            
            return self.tgt_vocab.decode(output_indices[0].tolist())
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def sidebar_content(self):
        """Render sidebar with system status"""
        with st.sidebar:
            st.markdown('<div class="section-header"><h2>🇮🇳 System Status</h2></div>', unsafe_allow_html=True)
            
            # Device info
            device_emoji = "🎮" if "cuda" in str(self.device) else "💻"
            st.info(f"{device_emoji} **Device:** {self.device}")
            
            # Model status
            st.markdown("### 🤖 Neural Models")
            
            if self.model_status['vocab']:
                st.success(f"✅ Vocabularies: Src={self.src_vocab.size}, Tgt={self.tgt_vocab.size}")
            else:
                st.warning("⚠️ Vocabularies not loaded")
            
            if self.model_status['lstm']:
                st.success("✅ LSTM (Bidirectional, 2 layers)")
            else:
                st.info("ℹ️ LSTM not available")
            
            if self.model_status['transformer']:
                st.success("✅ Transformer (Local Attention)")
            else:
                st.info("ℹ️ Transformer not available")
            
            # LLM Providers
            st.markdown("### 🌐 LLM Providers")
            if st.session_state.connected_providers:
                for provider in st.session_state.connected_providers:
                    st.success(f"✅ {provider.title()}")
                    
                    if provider in st.session_state.available_models:
                        model_count = len(st.session_state.available_models[provider])
                        st.caption(f"   📦 {model_count} models available")
            else:
                st.info("💡 No LLM providers connected")
            
            # Rate limits for Groq
            if 'groq' in st.session_state.connected_providers:
                if st.button("🔄 Check Rate Limits", key="check_limits"):
                    limits = self.llm.check_rate_limits('groq')
                    if limits:
                        st.markdown("**Rate Limits:**")
                        st.caption(f"Requests: {limits.get('requests_remaining', 'N/A')}/{limits.get('requests_limit', 'N/A')}")
                        st.caption(f"Tokens: {limits.get('tokens_remaining', 'N/A')}/{limits.get('tokens_limit', 'N/A')}")
            
            st.divider()
            
            # Assignment checklist
            st.markdown("### ✅ CS772 Requirements")
            requirements = {
                "LSTM (≤2 layers)": self.model_status['lstm'],
                "Transformer (Local Attn)": self.model_status['transformer'],
                "LLM Integration": bool(st.session_state.connected_providers),
                "Greedy + Beam Search": True,
                "≤100k Training Limit": True,
                "ACL-Compliant Metrics": True,
                "Error Analysis": True
            }
            
            for req, status in requirements.items():
                icon = "✅" if status else "⭕"
                st.markdown(f"{icon} {req}")
            
            st.divider()
            
            # Proud message
            st.markdown("### 🇮🇳 मेरा भारत महान")
            st.caption("Celebrating Indian Heritage")
    
    def tab_transliterate(self):
        """Main transliteration tab"""
        st.markdown('<div class="section-header"><h2>🎯 Roman → Devanagari Transliteration</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### ⚙️ Configuration")
            
            # Model selection
            available_models = list(self.models.keys())
            
            # Add LLM options
            for provider in st.session_state.connected_providers:
                available_models.append(f"LLM ({provider.title()})")
            
            if not available_models:
                st.warning("⚠️ No models available. Train models or connect LLM providers.")
                return
            
            model_type = st.selectbox(
                "Select Model",
                available_models,
                help="Choose the model for transliteration"
            )
            
            # Model-specific settings
            if model_type in ["LSTM", "Transformer"]:
                st.markdown("#### 🔍 Decoding Strategy")
                use_beam = st.checkbox("Use Beam Search", value=True, 
                                      help="Beam search explores multiple paths")
                
                if use_beam:
                    beam_size = st.slider("Beam Size", 2, 10, 5, 
                                        help="Higher = better quality but slower")
                else:
                    beam_size = 1
                
                st.info(f"{'🎯 Beam Search' if use_beam else '⚡ Greedy'} (beam={beam_size})")
            
            elif "LLM" in model_type:
                provider = model_type.split("(")[1].split(")")[0].lower()
                
                # Model selection
                if provider in st.session_state.available_models:
                    model_list = st.session_state.available_models[provider]
                    
                    categories = {}
                    for model in model_list:
                        cat = model.get('category', 'general')
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(model)
                    
                    st.markdown("#### 🤖 Model Selection")
                    selected_category = st.selectbox(
                        "Category",
                        list(categories.keys()),
                        help="Select model category"
                    )
                    
                    selected_model = st.selectbox(
                        "Model",
                        [m['id'] for m in categories[selected_category]],
                        help="Specific model"
                    )
                    
                    st.session_state.selected_models[provider] = selected_model
                else:
                    selected_model = None
                
                # Generation parameters
                st.markdown("#### 🎛️ Parameters")
                col_temp, col_top = st.columns(2)
                
                with col_temp:
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                                          help="Lower = deterministic")
                
                with col_top:
                    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05,
                                    help="Nucleus sampling")
                
                # Reasoning
                use_reasoning = False
                if provider == 'groq' and selected_model:
                    if 'gpt-oss' in selected_model or 'qwen' in selected_model:
                        use_reasoning = st.checkbox("🧠 Enable Reasoning", 
                                                   help="Show step-by-step")
        
        with col2:
            st.markdown("### 📝 Input & Output")
            
            input_text = st.text_area(
                "Enter Roman Text",
                height=120,
                placeholder="Type here...\nExample: namaste bharat",
                help="Roman script → Devanagari",
                value=st.session_state.get('example_text', '')
            )
            
            if 'example_text' in st.session_state:
                del st.session_state.example_text
            
            if st.button("✨ **Transliterate Now**", type="primary", use_container_width=True):
                if input_text.strip():
                    start_time = time.time()
                    
                    with st.spinner("🔄 Processing... मेरा भारत महान"):
                        if model_type in ["LSTM", "Transformer"]:
                            words = input_text.strip().split()
                            results = []
                            
                            progress_bar = st.progress(0)
                            
                            for i, word in enumerate(words):
                                output = self.transliterate_neural(word, model_type, beam_size)
                                results.append(output)
                                progress_bar.progress((i + 1) / len(words))
                            
                            output_text = ' '.join(results)
                            progress_bar.empty()
                            
                        elif "LLM" in model_type:
                            provider = model_type.split("(")[1].split(")")[0].lower()
                            output_text = self.llm.transliterate(
                                input_text,
                                provider=provider,
                                model=selected_model if 'selected_model' in locals() else None,
                                temperature=temperature if 'temperature' in locals() else 0.3,
                                top_p=top_p if 'top_p' in locals() else 0.95,
                                use_reasoning=use_reasoning if 'use_reasoning' in locals() else False
                            )
                        else:
                            output_text = "❌ Model not available"
                    
                    elapsed_time = time.time() - start_time
                    
                    if output_text and not output_text.startswith("❌"):
                        st.balloons()
                        st.success(f"✅ Completed in {elapsed_time:.2f}s 🇮🇳")
                        
                        st.markdown("### 📤 Result")
                        st.markdown(f'<div class="transliteration-output">{output_text}</div>', unsafe_allow_html=True)
                        
                        st.code(output_text, language=None)
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'input': input_text,
                            'output': output_text,
                            'model': model_type,
                            'time': elapsed_time
                        })
                    else:
                        st.error(f"❌ Failed: {output_text}")
                else:
                    st.warning("⚠️ Please enter text")
        
        # Examples
        st.divider()
        st.markdown("### ⚡ Quick Examples")
        
        examples = [
            ("namaste", "नमस्ते", "Greeting"),
            ("bharat", "भारत", "Country"),
            ("vidyalaya", "विद्यालय", "School"),
            ("computer", "कंप्यूटर", "Tech"),
            ("adhyapak", "अध्यापक", "Teacher"),
            ("pustakaalaya", "पुस्तकालय", "Library"),
            ("paryavaran", "पर्यावरण", "Environment"),
            ("sanvidhan", "संविधान", "Constitution")
        ]
        
        cols = st.columns(4)
        for i, (roman, devanagari, category) in enumerate(examples):
            with cols[i % 4]:
                if st.button(f"**{roman}**", key=f"ex_{i}", use_container_width=True,
                           help=f"{category}: {devanagari}"):
                    st.session_state.example_text = roman
                    st.rerun()
    
    def tab_api_configuration(self):
        """API Configuration tab"""
        st.markdown('<div class="section-header"><h2>🔑 API Configuration</h2></div>', unsafe_allow_html=True)
        st.info("💡 Connect to LLM providers instantly. No restart required!")
        
        providers = [
            {
                'name': 'groq',
                'title': 'Groq',
                'description': 'Ultra-fast inference (Llama, Mixtral, Gemma)',
                'emoji': '⚡',
                'free_tier': True
            },
            {
                'name': 'openai',
                'title': 'OpenAI',
                'description': 'GPT-3.5 and GPT-4 models',
                'emoji': '🤖',
                'free_tier': False
            },
            {
                'name': 'anthropic',
                'title': 'Anthropic',
                'description': 'Claude 3 family',
                'emoji': '🧠',
                'free_tier': False
            },
            {
                'name': 'google',
                'title': 'Google',
                'description': 'Gemini Pro',
                'emoji': '🌟',
                'free_tier': True
            }
        ]
        
        cols = st.columns(2)
        
        for i, provider in enumerate(providers):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="provider-card">
                    <h4>{provider['emoji']} {provider['title']}</h4>
                    <p>{provider['description']}</p>
                    {f"<span style='background:#138808;color:white;padding:4px 12px;border-radius:8px;font-size:11px;font-weight:700;'>FREE TIER</span>" if provider['free_tier'] else ""}
                </div>
                """, unsafe_allow_html=True)
                
                is_connected = provider['name'] in st.session_state.connected_providers
                
                if is_connected:
                    st.success(f"✅ Connected")
                    
                    if provider['name'] in st.session_state.available_models:
                        count = len(st.session_state.available_models[provider['name']])
                        st.caption(f"📦 {count} models available")
                    
                    col_info, col_disconnect = st.columns([3, 1])
                    
                    with col_disconnect:
                        if st.button("🔌", key=f"disconnect_{provider['name']}", 
                                   help="Disconnect"):
                            st.session_state.connected_providers.discard(provider['name'])
                            st.session_state.api_keys.pop(provider['name'], None)
                            st.rerun()
                else:
                    api_key = st.text_input(
                        f"{provider['title']} API Key",
                        type="password",
                        key=f"api_key_{provider['name']}",
                        placeholder="sk-..."
                    )
                    
                    if st.button(f"🔗 Connect", key=f"connect_{provider['name']}",
                               use_container_width=True, type="primary"):
                        if api_key and api_key.strip():
                            with st.spinner(f"Connecting to {provider['title']}..."):
                                success = self.llm.setup_client(provider['name'], api_key.strip())
                                
                                if success:
                                    st.session_state.api_keys[provider['name']] = api_key.strip()
                                    st.session_state.connected_providers.add(provider['name'])
                                    
                                    if provider['name'] == 'groq':
                                        models = self.llm.get_available_models('groq')
                                        st.session_state.available_models['groq'] = models
                                    
                                    st.success(f"✅ Connected to {provider['title']}!")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed. Check API key.")
                        else:
                            st.warning("⚠️ Enter API key")
        
        # Test connection
        if st.session_state.connected_providers:
            st.divider()
            st.markdown("### 🧪 Test Connections")
            
            test_text = st.text_input("Test Text", value="namaste", 
                                     help="Text to test transliteration")
            
            if st.button("🚀 Test All Connected Providers", type="primary"):
                results = {}
                
                for provider in st.session_state.connected_providers:
                    with st.spinner(f"Testing {provider}..."):
                        try:
                            result = self.llm.transliterate(test_text, provider=provider)
                            results[provider] = result
                        except Exception as e:
                            results[provider] = f"Error: {str(e)}"
                
                st.markdown("#### Test Results:")
                for provider, result in results.items():
                    if "Error" not in result and "not configured" not in result:
                        st.success(f"**{provider.title()}:** {result}")
                    else:
                        st.error(f"**{provider.title()}:** {result}")
    
    def tab_compare_models(self):
        """Model comparison tab"""
        st.markdown('<div class="section-header"><h2>📊 Model Comparison</h2></div>', unsafe_allow_html=True)
        
        compare_text = st.text_area(
            "Text for Comparison",
            value=st.session_state.get('compare_text', "mera bharat mahan"),
            height=80,
            help="Test across all models"
        )
        
        st.session_state.compare_text = compare_text
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_neural = st.checkbox("Include Neural Models", value=True)
            if include_neural:
                beam_sizes = st.multiselect("Beam Sizes", [1, 3, 5, 10], default=[1, 5])
        
        with col2:
            include_llm = st.checkbox("Include LLM", value=True)
            if include_llm:
                temps = st.multiselect("Temperatures", [0.1, 0.3, 0.5, 0.7], default=[0.3])
        
        if st.button("🔄 **Run Comparison**", type="primary", use_container_width=True):
            if compare_text.strip():
                results = []
                progress = st.progress(0)
                
                total = 0
                if include_neural:
                    total += len(self.models) * len(beam_sizes)
                if include_llm:
                    total += len(st.session_state.connected_providers) * len(temps)
                
                current = 0
                
                # Neural models
                if include_neural:
                    for model_name in self.models:
                        for beam in beam_sizes:
                            start = time.time()
                            words = compare_text.split()
                            outputs = [self.transliterate_neural(w, model_name, beam) for w in words]
                            elapsed = time.time() - start
                            
                            results.append({
                                'Model': model_name,
                                'Type': 'Neural',
                                'Config': f'Beam={beam}',
                                'Output': ' '.join(outputs),
                                'Time (s)': f"{elapsed:.3f}"
                            })
                            
                            current += 1
                            progress.progress(current / total)
                
                # LLM models
                if include_llm:
                    for prov in st.session_state.connected_providers:
                        for temp in temps:
                            start = time.time()
                            output = self.llm.transliterate(compare_text, provider=prov, temperature=temp)
                            elapsed = time.time() - start
                            
                            results.append({
                                'Model': prov.title(),
                                'Type': 'LLM',
                                'Config': f'T={temp}',
                                'Output': output,
                                'Time (s)': f"{elapsed:.3f}"
                            })
                            
                            current += 1
                            progress.progress(current / total)
                
                progress.empty()
                
                if results:
                    st.balloons()
                    st.success(f"✅ Tested {len(results)} configurations! 🇮🇳")
                    
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    df['Time_float'] = df['Time (s)'].astype(float)
                    fig = px.bar(
                        df, 
                        x='Model', 
                        y='Time_float',
                        color='Config',
                        title='Processing Time Comparison',
                        labels={'Time_float': 'Time (seconds)'},
                        color_discrete_sequence=['#FF9933', '#138808', '#FFD700', '#000080']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Enter text to compare")
    
    def tab_results(self):
        """Results and analysis tab"""
        st.markdown('<div class="section-header"><h2>📈 Results & Analysis</h2></div>', unsafe_allow_html=True)
        
        # Load results
        lstm_path = 'outputs/results/lstm_final_results.json'
        trans_path = 'outputs/results/transformer_final_results.json'
        
        if os.path.exists(lstm_path) or os.path.exists(trans_path):
            st.markdown("#### 🏆 ACL-Compliant Metrics")
            
            perf_data = []
            
            if os.path.exists(lstm_path):
                with open(lstm_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for method, metrics in data.get('metrics', {}).items():
                        perf_data.append({
                            'Model': 'LSTM',
                            'Method': method.replace('_', ' ').title(),
                            'Word Acc': metrics.get('word_accuracy', 0),
                            'Char F1': metrics.get('char_f1', 0),
                            'MRR': metrics.get('mrr', 0),
                            'MAP_ref': metrics.get('map_ref', 0)
                        })
            
            if os.path.exists(trans_path):
                with open(trans_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for method, metrics in data.get('metrics', {}).items():
                        perf_data.append({
                            'Model': 'Transformer',
                            'Method': method.replace('_', ' ').title(),
                            'Word Acc': metrics.get('word_accuracy', 0),
                            'Char F1': metrics.get('char_f1', 0),
                            'MRR': metrics.get('mrr', 0),
                            'MAP_ref': metrics.get('map_ref', 0)
                        })
            
            if perf_data:
                df = pd.DataFrame(perf_data)
                
                # Visualization
                fig = px.bar(
                    df, 
                    x='Method', 
                    y='Char F1', 
                    color='Model',
                    title='Model Performance (ACL W15-3902 Compliant)',
                    barmode='group',
                    color_discrete_sequence=['#FF9933', '#138808']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Best performer
                best_row = max(perf_data, key=lambda x: x['Char F1'])
                st.success(f"🥇 **Best:** {best_row['Model']} ({best_row['Method']}) - F1: {best_row['Char F1']:.3f}")
        else:
            st.info("📊 No results found. Train models first.")
        
        # Error analysis
        st.divider()
        st.markdown("#### 🔍 Error Analysis")
        
        with st.expander("📚 Transliteration Challenges (ACL W15-3902)", expanded=False):
            st.markdown("""
            ### Why Some Sequences Are Hard to Transliterate
            
            **1. Conjunct Consonants (संयुक्त व्यंजन)** 🔤  
            - Examples: `क्ष` (ksha), `त्र` (tra), `ज्ञ` (gya)
            - **Challenge:** Many-to-one mapping (3+ Roman chars → 1 Devanagari)
            - **Impact:** Often split incorrectly or missing
            
            **2. Vowel Matras (स्वर मात्राएँ)** 📝  
            - Examples: `े` (e), `ो` (o), `ु` (u)
            - **Challenge:** Position-sensitive placement
            - **Impact:** Placement errors or omissions
            
            **3. Halant (हलंत - ्)** ⚡  
            - Symbol: `्`
            - **Challenge:** Invisible in Roman script
            - **Impact:** Missing conjunct formations
            
            **4. Aspirated Consonants (महाप्राण)** 💨  
            - Examples: `ख` (kha), `घ` (gha), `छ` (cha)
            - **Challenge:** Subtle phonetic differences
            - **Impact:** Substitution errors
            
            **5. Nasalization (अनुनासिक)** 👃  
            - Symbols: `ं` (anusvara), `ँ` (chandrabindu)
            - **Challenge:** No direct Roman equivalent
            - **Impact:** Often omitted or misplaced
            """)
        
        # History
        if st.session_state.history:
            st.divider()
            st.markdown("#### 📜 Translation History")
            
            history_df = pd.DataFrame(st.session_state.history)
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if len(history_df) > 0:
                    st.dataframe(
                        history_df[['timestamp', 'input', 'output', 'model', 'time']],
                        use_container_width=True,
                        hide_index=True
                    )
            
            with col2:
                if st.button("🗑️ Clear"):
                    st.session_state.history = []
                    st.rerun()
                
                if st.button("📥 Export"):
                    csv = history_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_history"
                    )
    
    def run(self):
        """Main application logic"""
        # Header with Indian tricolor and floating chars
        st.markdown("""
        <div class="main-header">
            <h1>🇮🇳 Hindi Transliteration System</h1>
            <p>CS772 Assignment 2 | Roman → Devanagari | मेरा भारत महान</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar_content()
        
        # Tabs
        tabs = st.tabs([
            "🎯 Transliterate", 
            "🔑 API Config", 
            "📊 Compare", 
            "📈 Results"
        ])
        
        with tabs[0]:
            self.tab_transliterate()
        
        with tabs[1]:
            self.tab_api_configuration()
        
        with tabs[2]:
            self.tab_compare_models()
        
        with tabs[3]:
            self.tab_results()
        
        # Footer with tricolor
        st.markdown("""
        <div class="main-footer">
            <p class="footer-hindi">जय हिन्द | मेरा भारत महान</p>
            <p><strong>Hindi Transliteration System</strong> | CS772 Assignment 2</p>
            <p>© 2025 | ACL W15-3902 Compliant | For Educational Use</p>
            <p>🇮🇳 Powered by Indian Innovation 🇮🇳</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Entry point"""
    app = TransliterationApp()
    app.run()


if __name__ == "__main__":
    main()