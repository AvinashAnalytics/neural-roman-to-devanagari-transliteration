"""
LLM-based Transliteration Model for CS772 Assignment 2
Roman to Devanagari using OpenAI, Anthropic, Google Gemini, Groq, DeepInfra
Config-driven with systematic evaluation and caching
"""

import os
import json
import yaml
import time
import re
import requests
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LLM libraries (all optional)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class LLMTransliterator:
    """
    LLM-based transliterator with config-driven parameters.
    Supports multiple providers and systematic evaluation.
    """
    
    def __init__(self, config: Union[Dict, str, Path]):
        """
        Initialize LLM transliterator.
        
        Args:
            config: Config dict or path to config.yaml
        """
        # Load config if path provided
        if isinstance(config, (str, Path)):
            with open(config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.llm_config = config.get('llm', {})
        
        # Extract config parameters
        self.max_tokens = self.llm_config.get('max_tokens', 100)
        self.timeout = self.llm_config.get('timeout', 30)
        self.max_retries = self.llm_config.get('max_retries', 3)
        self.retry_delay = self.llm_config.get('retry_delay', 1.0)
        self.requests_per_minute = self.llm_config.get('requests_per_minute', 60)
        
        # Prompts from config
        self.system_prompt = self.llm_config.get(
            'system_prompt',
            'You are a Hindi transliteration expert. Convert Roman script to Devanagari script. '
            'Provide ONLY the Devanagari transliteration, no explanations.'
        )
        self.user_prompt_template = self.llm_config.get(
            'user_prompt_template',
            'Transliterate to Devanagari: {roman_text}'
        )
        
        # API key environment variables
        self.api_key_env_vars = self.llm_config.get('api_key_env_vars', {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'groq': 'GROQ_API_KEY',
            'deepinfra': 'DEEPINFRA_API_KEY'
        })
        
        # State
        self.clients = {}
        self.api_keys = {}
        self.rate_limiters = {}
        self.model_cache = {}
        self.cache_expiry = {}
        
        # Results directory
        self.results_dir = Path(config.get('paths', {}).get('results_dir', 'outputs/results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("LLM Transliterator initialized")
    
    def setup_client(self, provider: str, api_key: Optional[str] = None) -> bool:
        """
        Setup API client with validation.
        
        Args:
            provider: Provider name (openai, anthropic, google, groq, deepinfra)
            api_key: API key (if None, reads from env var)
        
        Returns:
            True if setup successful
        """
        provider = provider.lower()
        
        # Get API key from env if not provided
        if api_key is None:
            env_var = self.api_key_env_vars.get(provider)
            if env_var:
                api_key = os.environ.get(env_var)
            if not api_key:
                logger.warning(f"No API key found for {provider} (env var: {env_var})")
                return False
        
        try:
            if provider == 'openai' and openai:
                if hasattr(openai, 'OpenAI'):
                    # New API (>= 1.0)
                    self.clients['openai'] = openai.OpenAI(api_key=api_key)
                else:
                    # Old API (< 1.0)
                    openai.api_key = api_key
                    self.clients['openai'] = openai
                
                self.api_keys['openai'] = api_key
                self.rate_limiters['openai'] = RateLimiter(self.requests_per_minute)
                logger.info("✅ OpenAI client configured")
                return True
            
            elif provider == 'anthropic' and anthropic:
                self.clients['anthropic'] = anthropic.Anthropic(api_key=api_key)
                self.api_keys['anthropic'] = api_key
                self.rate_limiters['anthropic'] = RateLimiter(self.requests_per_minute)
                logger.info("✅ Anthropic client configured")
                return True
            
            elif provider == 'google' and genai:
                genai.configure(api_key=api_key)
                self.clients['google'] = genai.GenerativeModel('gemini-pro')
                self.api_keys['google'] = api_key
                self.rate_limiters['google'] = RateLimiter(self.requests_per_minute)
                logger.info("✅ Google Gemini client configured")
                return True
            
            elif provider == 'groq':
                if Groq:
                    self.clients['groq'] = Groq(api_key=api_key)
                elif openai and hasattr(openai, 'OpenAI'):
                    # Fallback: use OpenAI client with Groq base URL
                    self.clients['groq'] = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                else:
                    # Fallback: use requests
                    self.clients['groq'] = {'api_key': api_key, 'use_requests': True}
                
                self.api_keys['groq'] = api_key
                self.rate_limiters['groq'] = RateLimiter(self.requests_per_minute)
                
                # Fetch available models
                self.fetch_groq_models()
                logger.info("✅ Groq client configured")
                return True
            
            elif provider == 'deepinfra':
                self.clients['deepinfra'] = {'api_key': api_key}
                self.api_keys['deepinfra'] = api_key
                self.rate_limiters['deepinfra'] = RateLimiter(self.requests_per_minute)
                logger.info("✅ DeepInfra client configured")
                return True
            
            else:
                logger.error(f"Provider {provider} not supported or library not installed")
                return False
        
        except Exception as e:
            logger.error(f"Error setting up {provider}: {e}")
            return False
    
    def fetch_groq_models(self) -> List[Dict]:
        """Fetch available Groq models with caching"""
        try:
            # Check cache (1 hour TTL)
            if 'groq' in self.model_cache:
                if datetime.now() < self.cache_expiry.get('groq', datetime.min):
                    return self.model_cache['groq']
            
            headers = {
                "Authorization": f"Bearer {self.api_keys['groq']}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                models_data = response.json()
                models = []
                
                for model in models_data.get('data', []):
                    model_id = model.get('id', '')
                    # Filter out deprecated/special models
                    if 'whisper' not in model_id.lower() and 'deprecated' not in model_id.lower():
                        models.append({
                            'id': model_id,
                            'name': model.get('name', model_id),
                            'context_window': model.get('context_window', 0)
                        })
                
                # Cache for 1 hour
                self.model_cache['groq'] = models
                self.cache_expiry['groq'] = datetime.now() + timedelta(hours=1)
                
                # Limit cache size
                if len(self.model_cache) > 10:
                    oldest_key = min(self.cache_expiry, key=self.cache_expiry.get)
                    del self.model_cache[oldest_key]
                    del self.cache_expiry[oldest_key]
                
                return models
            else:
                logger.warning(f"Failed to fetch Groq models: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching Groq models: {e}")
            return []
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response to extract Devanagari text.
        Preserves valid Devanagari punctuation.
        """
        if not response:
            return ""
        
        # Remove common prefixes/suffixes
        response = re.sub(
            r'^(Devanagari:|Output:|Result:|Translation:)\s*',
            '', response.strip(), flags=re.IGNORECASE
        )
        response = re.sub(
            r'\s*(This is the transliteration|Hope this helps).*$',
            '', response, flags=re.IGNORECASE
        )
        
        # Remove quotes
        response = response.strip('"\'`')
        
        # Extract first line with Devanagari
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if re.search(r'[\u0900-\u097F]', line):
                # Keep Devanagari characters, spaces, and Devanagari punctuation
                cleaned = re.sub(r'[^\u0900-\u097F\s।॥]', '', line)
                return cleaned.strip()
        
        return response.strip()
    
    def create_prompt(self, text: str) -> str:
        """Create prompt from config template"""
        # Sanitize input
        text = text.strip()
        if not text:
            return ""
        
        # Use config template
        return self.user_prompt_template.format(roman_text=text)
    
    def transliterate(
        self,
        text: str,
        provider: str = None,
        model: str = None,
        temperature: float = 0.3,
        top_p: float = 0.95
    ) -> str:
        """
        Transliterate text using specified LLM provider.
        
        Args:
            text: Roman text to transliterate
            provider: LLM provider (openai, anthropic, google, groq, deepinfra)
            model: Model name (provider-specific)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Devanagari transliteration
        """
        # Clean input
        text = text.strip()
        if not text:
            return ""
        
        # Select provider
        if provider is None:
            if self.clients:
                provider = list(self.clients.keys())[0]
            else:
                return "ERROR: No LLM provider configured"
        
        provider = provider.lower()
        
        if provider not in self.clients:
            return f"ERROR: Provider {provider} not configured"
        
        # Apply rate limiting
        if provider in self.rate_limiters:
            self.rate_limiters[provider].wait_if_needed()
        
        # Route to appropriate method with retry logic
        try:
            if provider == 'openai':
                return self._retry_with_backoff(
                    self._transliterate_openai, text, model, temperature, top_p
                )
            elif provider == 'anthropic':
                return self._retry_with_backoff(
                    self._transliterate_anthropic, text, model, temperature, top_p
                )
            elif provider == 'google':
                return self._retry_with_backoff(
                    self._transliterate_google, text, temperature, top_p
                )
            elif provider == 'groq':
                return self._retry_with_backoff(
                    self._transliterate_groq, text, model, temperature, top_p
                )
            elif provider == 'deepinfra':
                return self._retry_with_backoff(
                    self._transliterate_deepinfra, text, model, temperature, top_p
                )
            else:
                return f"ERROR: Provider {provider} not supported"
        
        except Exception as e:
            logger.error(f"Transliteration failed: {e}")
            return f"ERROR: {str(e)}"
    
    def _transliterate_openai(
        self, text: str, model: Optional[str], temperature: float, top_p: float
    ) -> str:
        """Transliterate using OpenAI"""
        client = self.clients['openai']
        
        if model is None:
            model = self.llm_config.get('default_model', 'gpt-3.5-turbo')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.create_prompt(text)}
        ]
        
        # Check API version
        if hasattr(client, 'chat'):
            # New API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            result = response.choices[0].message.content
        else:
            # Old API
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                request_timeout=self.timeout
            )
            result = response.choices[0].message['content']
        
        return self._clean_response(result)
    
    def _transliterate_anthropic(
        self, text: str, model: Optional[str], temperature: float, top_p: float
    ) -> str:
        """Transliterate using Anthropic Claude"""
        client = self.clients['anthropic']
        
        if model is None:
            model = 'claude-3-5-sonnet-20241022'
        
        message = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": self.create_prompt(text)}],
            timeout=self.timeout
        )
        
        result = message.content[0].text
        return self._clean_response(result)
    
    def _transliterate_google(
        self, text: str, temperature: float, top_p: float
    ) -> str:
        """Transliterate using Google Gemini"""
        model = self.clients['google']
        
        response = model.generate_content(
            self.create_prompt(text),
            generation_config={
                'temperature': temperature,
                'top_p': top_p,
                'max_output_tokens': self.max_tokens
            }
        )
        
        result = response.text
        return self._clean_response(result)
    
    def _transliterate_groq(
        self, text: str, model: Optional[str], temperature: float, top_p: float
    ) -> str:
        """Transliterate using Groq"""
        client = self.clients['groq']
        
        # Auto-select model if not specified
        if model is None:
            models = self.fetch_groq_models()
            model = models[0]['id'] if models else 'mixtral-8x7b-32768'
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.create_prompt(text)}
        ]
        
        # Check client type
        if isinstance(client, dict) and client.get('use_requests'):
            # Fallback: use requests
            return self._groq_request(text, model, temperature, top_p)
        else:
            # Use Groq/OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            result = response.choices[0].message.content
            return self._clean_response(result)
    
    def _groq_request(
        self, text: str, model: str, temperature: float, top_p: float
    ) -> str:
        """Fallback Groq API using requests"""
        headers = {
            "Authorization": f"Bearer {self.api_keys['groq']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.create_prompt(text)}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return self._clean_response(result)
    
    def _transliterate_deepinfra(
        self, text: str, model: Optional[str], temperature: float, top_p: float
    ) -> str:
        """Transliterate using DeepInfra"""
        if model is None:
            model = 'meta-llama/Llama-2-70b-chat-hf'
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['deepinfra']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.create_prompt(text)}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return self._clean_response(result)
    
    def batch_transliterate(
        self,
        texts: List[str],
        provider: str = None,
        model: str = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        show_progress: bool = True
    ) -> List[str]:
        """
        Batch transliteration with progress tracking.
        
        Args:
            texts: List of Roman texts
            provider: LLM provider
            model: Model name
            temperature: Sampling temperature
            top_p: Nucleus sampling
            show_progress: Show progress bar
        
        Returns:
            List of Devanagari translations
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Transliterating")
            except ImportError:
                iterator = texts
        else:
            iterator = texts
        
        for text in iterator:
            result = self.transliterate(text, provider, model, temperature, top_p)
            results.append(result)
        
        return results
    
    def experiment_temperature_top_p(
        self,
        texts: List[str],
        provider: str,
        model: str = None
    ) -> Dict:
        """
        Experiment with different temperature and top_p values (assignment requirement).
        
        Args:
            texts: List of test texts
            provider: LLM provider
            model: Model name
        
        Returns:
            Dictionary of results for each (temperature, top_p) combination
        """
        logger.info(f"Running temperature/top_p experiments for {provider}")
        
        temperature_values = self.llm_config.get('temperature_values', [0.1, 0.3, 0.5, 0.7])
        top_p_values = self.llm_config.get('top_p_values', [0.9, 0.95, 1.0])
        
        results = {}
        
        for temp in temperature_values:
            for top_p in top_p_values:
                key = f"temp_{temp}_top_p_{top_p}"
                logger.info(f"  Testing {key}...")
                
                predictions = []
                for text in texts:
                    pred = self.transliterate(text, provider, model, temp, top_p)
                    predictions.append(pred)
                
                results[key] = {
                    'temperature': temp,
                    'top_p': top_p,
                    'predictions': predictions
                }
        
        return results
    
    def save_results(
        self,
        results: Dict,
        provider: str,
        model: str,
        metadata: Optional[Dict] = None
    ):
        """Save experiment results to JSON"""
        output = {
            'provider': provider,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'metadata': metadata or {}
        }
        
        filename = f'llm_{provider}_{model.replace("/", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        filepath = self.results_dir / filename
        
        # Atomic write
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        temp_path.replace(filepath)
        
        logger.info(f"💾 Results saved: {filepath}")


def main():
    """Test LLM transliterator"""
    print("🧪 Testing LLM Transliterator\n")
    
    # Load config
    config_path = "config/config.yaml"
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        return
    
    llm = LLMTransliterator(config_path)
    
    # Test Groq (free tier available)
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        print("Testing Groq...")
        success = llm.setup_client('groq')
        
        if success:
            # Fetch models
            models = llm.fetch_groq_models()
            print(f"✅ Available models: {len(models)}")
            for model in models[:5]:
                print(f"   - {model['id']}")
            
            # Test transliteration
            test_cases = ["namaste", "bharat", "computer"]
            
            print("\nTesting transliteration:")
            for text in test_cases:
                result = llm.transliterate(text, provider='groq')
                print(f"  {text} → {result}")
            
            # Temperature experiment
            print("\nTesting temperature variation:")
            for temp in [0.1, 0.5, 0.9]:
                result = llm.transliterate("namaste", provider='groq', temperature=temp)
                print(f"  temp={temp}: {result}")
    else:
        print("⚠️  No GROQ_API_KEY in environment")
        print("Set with: export GROQ_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()