import os
import psycopg2
import psycopg2.pool
import json
import math
import requests
import hashlib
import random
import asyncio
import time
import threading
from typing import List, Dict, Optional, Tuple
from datetime import timedelta, datetime, timezone
import discord
from discord import app_commands
from discord.ext import commands, tasks
LEARNING_CHANNEL = '1411335494234669076'
TRADING_CHANNEL = '1418976581099065355'
AUTO_KICK_CHANNEL = '1411335541873709167'
HELPER_ROLES = ['1418434355650625676', '1352853011424219158', '1372300233240739920']
STUDENT_ROLE = '1341949236471926805'
OWNER_ID = '1334138321412296725'
PORT = int(os.getenv('PORT', 5000))

TOKEN = os.getenv('TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

user_messages = {}
user_message_timestamps = {}
roast_chance = 0.5

KB_REPLY_CONFIDENCE = 0.85
KB_SUGGEST_CONFIDENCE = 0.75

pending_conversations = {}
conversation_lock = threading.Lock()

KB_REPLY_MODE = "You are Bloom — concise assistant. Rephrase this helper answer in 1 sentence, 6-12 short words. Plain vocabulary only. No emojis, code blocks, or markdown."
AI_FALLBACK_MODE = "You are Bloom — clear, critical, and educational. Give SHORT answer (1-2 sentences, 15-35 words). Question assumptions. Avoid praise and fluff."

def compute_student_points(query: str, question: str) -> float:
    """Calculate student points based on lexical hits, intent match, and specificity."""
    import re

    query_lower = query.lower().strip()
    question_lower = question.lower().strip()

    question_words = set(re.findall(r'\b\w+\b', question_lower))
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    if not question_words:
        lexical_hits = 0.0
    else:
        matching_words = question_words.intersection(query_words)
        lexical_hits = len(matching_words) / len(question_words)

    intent_words = {'how', 'what', 'why', 'when', 'where', 'who', 'which'}
    query_has_intent = any(word in query_words for word in intent_words)
    question_has_intent = any(word in question_words for word in intent_words)
    intent_match = 1.0 if query_has_intent and question_has_intent else 0.0

    has_numbers = bool(re.search(r'\d', query_lower))
    length_score = min(len(query_lower) / 100.0, 1.0)
    specificity = (0.6 * length_score + 0.4 * (1.0 if has_numbers else 0.0))

    student_points = 0.4 * lexical_hits + 0.35 * intent_match + 0.25 * specificity
    return student_points

def compute_recency_score(created_at: str) -> float:
    """Calculate recency score based on age of knowledge entry."""
    try:
        if 'Z' in created_at or '+' in created_at:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            created_dt = datetime.fromisoformat(created_at)
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - created_dt).days
        recency_score = max(0.0, 1.0 - age_days / 365.0)
        return recency_score
    except Exception as e:
        print(f"⚠️  Warning: Failed to compute recency score for timestamp '{created_at}': {e}")
        return 0.5

def combined_confidence(semantic_sim: float, student_points: float, recency_score: float) -> float:
    """Calculate combined confidence score with weighted components."""
    w1, w2, w3 = 0.60, 0.25, 0.15
    return w1 * semantic_sim + w2 * student_points + w3 * recency_score

def detect_pii(text: str) -> bool:
    """Detect PII (emails, phone numbers, addresses) in text."""
    import re

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, text):
        return True

    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    if re.search(phone_pattern, text):
        return True

    address_pattern = r'\d+\s+[\w\s]+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir|way)'
    if re.search(address_pattern, text, re.IGNORECASE):
        return True

    return False

def calculate_q_clear(question: str) -> float:
    """Calculate question clarity score based on length, structure, and question words."""
    import re

    question = question.strip()
    if not question:
        return 0.0

    words = re.findall(r'\b\w+\b', question.lower())
    if not words:
        return 0.0

    length_score = min(len(words) / 15.0, 1.0)

    question_words = {'how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'is', 'are', 'do', 'does'}
    has_question_word = any(word in words for word in question_words)
    question_word_score = 1.0 if has_question_word else 0.5

    has_punctuation = '?' in question or '.' in question
    punctuation_score = 1.0 if has_punctuation else 0.8

    q_clear = 0.5 * length_score + 0.3 * question_word_score + 0.2 * punctuation_score
    return min(q_clear, 1.0)

def calculate_a_substance(answer: str) -> float:
    """Calculate answer substance score based on length, detail, and informativeness."""
    import re

    answer = answer.strip()
    if not answer:
        return 0.0

    words = re.findall(r'\b\w+\b', answer.lower())
    if not words:
        return 0.0

    length_score = min(len(words) / 25.0, 1.0)

    filler_words = {'um', 'uh', 'like', 'just', 'really', 'very', 'actually', 'basically'}
    filler_count = sum(1 for word in words if word in filler_words)
    filler_penalty = max(0, 1.0 - (filler_count / len(words)) * 2)

    has_details = len(words) > 10 and (bool(re.search(r'\d', answer)) or any(w in words for w in ['because', 'since', 'therefore', 'thus', 'due']))
    detail_score = 1.0 if has_details else 0.6

    a_substance = 0.5 * length_score + 0.3 * filler_penalty + 0.2 * detail_score
    return min(a_substance, 1.0)

def is_greeting_or_casual(text: str) -> bool:
    """Detect if message is a greeting or casual conversation."""
    import re

    text_lower = text.lower().strip()
    words = re.findall(r'\b\w+\b', text_lower)

    if len(words) <= 3:
        greetings = {
            'hi', 'hello', 'hey', 'yo', 'sup', 'what\'s up', 'whats up',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'thanks', 'thank you', 'ty', 'thx', 'ok', 'okay', 'sure',
            'yes', 'no', 'yeah', 'yep', 'nope', 'nah', 'lol', 'lmao',
            'brb', 'gtg', 'afk', 'dm me', 'check dm', 'check dms'
        }

        for greeting in greetings:
            if greeting in text_lower:
                return True

    casual_patterns = [
        r'^(hi|hello|hey|yo|sup|wassup)\s*$',
        r'^(thanks|thank you|ty|thx)\s*$',
        r'^(ok|okay|sure|cool|nice)\s*$',
        r'^(lol|lmao|haha|hehe)\s*$',
        r'^(brb|gtg|afk)\s*$',
        r'^\w{1,3}$'
    ]

    for pattern in casual_patterns:
        if re.match(pattern, text_lower):
            return True

    return False

def is_blabberish(text: str) -> bool:
    """Detect if text is blabberish or nonsensical."""
    import re

    text = text.strip()
    if not text or len(text) < 3:
        return True

    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return True

    if len(set(words)) == 1 and len(words) > 2:
        return True

    consonant_heavy = sum(1 for char in text.lower() if char in 'bcdfghjklmnpqrstvwxyz')
    vowel_count = sum(1 for char in text.lower() if char in 'aeiou')

    if vowel_count > 0:
        consonant_vowel_ratio = consonant_heavy / vowel_count
        if consonant_vowel_ratio > 5:
            return True

    repeated_chars = re.findall(r'(.)\1{4,}', text)
    if repeated_chars:
        return True

    return False

def is_teaching_content(text: str) -> bool:
    """Determine if text contains teaching/educational content."""
    import re

    text_lower = text.lower().strip()
    words = re.findall(r'\b\w+\b', text_lower)

    if len(words) < 3:
        return False

    instructional_verbs = {
        'install', 'run', 'use', 'open', 'close', 'start', 'stop', 'restart',
        'click', 'press', 'type', 'enter', 'select', 'choose', 'delete', 'remove',
        'add', 'create', 'make', 'build', 'set', 'configure', 'change', 'modify',
        'update', 'upgrade', 'download', 'upload', 'save', 'load', 'import', 'export',
        'copy', 'paste', 'cut', 'move', 'drag', 'drop', 'enable', 'disable',
        'activate', 'deactivate', 'check', 'uncheck', 'navigate', 'go to', 'try'
    }

    teaching_indicators = {
        'because', 'since', 'therefore', 'thus', 'however', 'although',
        'means', 'refers', 'indicates', 'shows', 'demonstrates',
        'example', 'such as', 'like', 'including', 'specifically',
        'first', 'second', 'then', 'next', 'finally', 'steps',
        'should', 'need to', 'must', 'have to', 'can', 'will', 'would'
    }

    has_instructional_verb = any(verb in text_lower for verb in instructional_verbs)
    if has_instructional_verb:
        return True

    has_code_or_command = bool(re.search(r'`[^`]+`|```', text))
    if has_code_or_command and len(words) >= 3:
        return True

    indicator_count = sum(1 for indicator in teaching_indicators if indicator in text_lower)
    if indicator_count >= 1:
        return True

    has_explanation_structure = any([
        ' is ' in text_lower and len(words) >= 5,
        ' are ' in text_lower and len(words) >= 5,
        re.search(r'\d+\s*(steps?|ways?|methods?|points?)', text_lower),
        re.search(r'(you|we)\s+(can|should|need|must|have to|will|would)', text_lower),
        re.search(r'(it|this|that)\s+(is|will|would|can|should)', text_lower)
    ])

    if has_explanation_structure:
        return True

    return False

def aggregate_messages(messages: List[Dict]) -> str:
    """Aggregate multiple messages into a coherent answer."""
    if not messages:
        return ""

    if len(messages) == 1:
        return messages[0]['content'].strip()

    filtered_messages = []
    for msg in messages:
        content = msg['content'].strip()
        if not is_greeting_or_casual(content) and not is_blabberish(content):
            filtered_messages.append(content)

    if not filtered_messages:
        return ""

    aggregated = ' '.join(filtered_messages)

    aggregated = ' '.join(aggregated.split())

    return aggregated

class Database:
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set")

        self.db_url = db_url
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=db_url
        )
        self.create_tables()

    def get_connection(self):
        return self.connection_pool.getconn()

    def put_connection(self, conn):
        self.connection_pool.putconn(conn)

    def create_tables(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL UNIQUE,
                        answer TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        q_clear REAL DEFAULT 0.0,
                        a_substance REAL DEFAULT 0.0,
                        approved INTEGER DEFAULT 0
                    )
                ''')

                cur.execute('''
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='knowledge' AND column_name='q_clear') THEN
                            ALTER TABLE knowledge ADD COLUMN q_clear REAL DEFAULT 0.0;
                        END IF;
                    END $$;
                ''')

                cur.execute('''
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='knowledge' AND column_name='a_substance') THEN
                            ALTER TABLE knowledge ADD COLUMN a_substance REAL DEFAULT 0.0;
                        END IF;
                    END $$;
                ''')

                cur.execute('''
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='knowledge' AND column_name='approved') THEN
                            ALTER TABLE knowledge ADD COLUMN approved INTEGER DEFAULT 0;
                        END IF;
                    END $$;
                ''')

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS ai_response_cache (
                        id SERIAL PRIMARY KEY,
                        question_hash TEXT NOT NULL UNIQUE,
                        question TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER DEFAULT 1
                    )
                ''')

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS setup_messages (
                        id SERIAL PRIMARY KEY,
                        channel_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
        finally:
            self.put_connection(conn)

    def save_knowledge(self, question: str, answer: str, embedding: List[float]) -> bool:
        try:
            if detect_pii(question) or detect_pii(answer):
                print(f"⚠️ PII detected, not storing: '{question[:30]}...'")
                return False

            q_clear = calculate_q_clear(question)
            a_substance = calculate_a_substance(answer)

            if q_clear < 0.6:
                print(f"⚠️ Low question clarity ({q_clear:.2f}), not storing: '{question[:30]}...'")
                return False

            if a_substance < 0.6:
                print(f"⚠️ Low answer substance ({a_substance:.2f}), not storing: '{question[:30]}...'")
                return False

            duplicate = self.find_similar_question(embedding, threshold=0.92)
            if duplicate:
                print(f"⚠️ Duplicate question detected (similarity: {duplicate['similarity']:.2f}), not storing: '{question[:30]}...'")
                return False

            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        'INSERT INTO knowledge (question, answer, embedding, q_clear, a_substance, approved) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (question) DO NOTHING',
                        (question, answer, json.dumps(embedding), q_clear, a_substance, 1)
                    )
                    conn.commit()
                    return cur.rowcount > 0
            finally:
                self.put_connection(conn)
        except Exception as e:
            print(f"Error saving knowledge: {e}")
            return False

    def get_all_knowledge(self) -> List[Dict]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM knowledge')
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        finally:
            self.put_connection(conn)

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    def find_similar_question(self, question_embedding: List[float], threshold: float = 0.65) -> Optional[Dict]:
        knowledge = self.get_all_knowledge()
        best_match = None
        best_score = 0

        for item in knowledge:
            stored_embedding = json.loads(item['embedding'])
            similarity = self.cosine_similarity(question_embedding, stored_embedding)

            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = {**item, 'similarity': similarity}

        return best_match

    def find_top_matches_with_confidence(self, query: str, question_embedding: List[float], top_n: int = 3, min_semantic: float = 0.65) -> List[Dict]:
        knowledge = self.get_all_knowledge()
        matches = []

        for item in knowledge:
            stored_embedding = json.loads(item['embedding'])
            semantic_sim = self.cosine_similarity(question_embedding, stored_embedding)

            if semantic_sim >= min_semantic:
                student_points = compute_student_points(query, item['question'])
                recency_score = compute_recency_score(item.get('created_at', ''))
                combined_conf = combined_confidence(semantic_sim, student_points, recency_score)

                matches.append({
                    **item,
                    'semantic_sim': semantic_sim,
                    'student_points': student_points,
                    'recency_score': recency_score,
                    'combined_confidence': combined_conf
                })

        matches.sort(key=lambda x: x['combined_confidence'], reverse=True)
        return matches[:top_n]

    def save_conversation(self, user_id: str, message: str, response: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO conversations (user_id, message, response) VALUES (%s, %s, %s)',
                    (user_id, message, response)
                )
                conn.commit()
        finally:
            self.put_connection(conn)

    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT message, response FROM conversations WHERE user_id = %s ORDER BY created_at DESC LIMIT %s',
                    (user_id, limit)
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return list(reversed([dict(zip(columns, row)) for row in rows]))
        finally:
            self.put_connection(conn)

    def _cleanup_old_conversations(self, user_id: str, max_per_user: int = 10):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT COUNT(*) as count FROM conversations WHERE user_id = %s',
                    (user_id,)
                )
                count = cur.fetchone()[0]

                if count > max_per_user:
                    cur.execute(
                        '''DELETE FROM conversations WHERE id IN (
                            SELECT id FROM conversations WHERE user_id = %s
                            ORDER BY created_at ASC LIMIT %s
                        )''',
                        (user_id, count - max_per_user)
                    )
                    conn.commit()
                    print(f"🧹 Cleaned up {count - max_per_user} old conversations for user {user_id}")
        finally:
            self.put_connection(conn)

    def cleanup_old_data(self, days: int = 30):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversations WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => %s)",
                    (days,)
                )
                deleted_conversations = cur.rowcount
                conn.commit()

                if deleted_conversations > 0:
                    print(f"🧹 Database cleanup: Removed {deleted_conversations} conversations older than {days} days")

                return deleted_conversations
        finally:
            self.put_connection(conn)

    def get_database_stats(self) -> Dict:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                stats = {}

                cur.execute('SELECT COUNT(*) as count FROM knowledge')
                stats['knowledge_count'] = cur.fetchone()[0]

                cur.execute('SELECT COUNT(*) as count FROM conversations')
                stats['conversations_count'] = cur.fetchone()[0]

                cur.execute('SELECT COUNT(DISTINCT user_id) as count FROM conversations')
                stats['unique_users'] = cur.fetchone()[0]

                try:
                    cur.execute('''
                        SELECT 
                            AVG(q_clear) as avg_q_clear,
                            AVG(a_substance) as avg_a_substance,
                            COUNT(CASE WHEN q_clear >= 0.8 THEN 1 END) as high_q_clear,
                            COUNT(CASE WHEN a_substance >= 0.8 THEN 1 END) as high_a_substance,
                            COUNT(CASE WHEN q_clear >= 0.6 AND q_clear < 0.8 THEN 1 END) as medium_q_clear,
                            COUNT(CASE WHEN a_substance >= 0.6 AND a_substance < 0.8 THEN 1 END) as medium_a_substance
                        FROM knowledge
                    ''')
                    quality_stats = cur.fetchone()
                    stats['avg_q_clear'] = quality_stats[0] or 0
                    stats['avg_a_substance'] = quality_stats[1] or 0
                    stats['high_q_clear'] = quality_stats[2] or 0
                    stats['high_a_substance'] = quality_stats[3] or 0
                    stats['medium_q_clear'] = quality_stats[4] or 0
                    stats['medium_a_substance'] = quality_stats[5] or 0
                except:
                    pass

                return stats
        finally:
            self.put_connection(conn)

    def export_knowledge_as_json(self) -> str:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, question, answer, created_at, q_clear, a_substance, approved FROM knowledge')
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                entries = [dict(zip(columns, row)) for row in rows]
                return json.dumps(entries, indent=2, default=str)
        finally:
            self.put_connection(conn)

    def purge_knowledge_by_id(self, entry_id: int) -> bool:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM knowledge WHERE id = %s', (entry_id,))
                conn.commit()
                return cur.rowcount > 0
        finally:
            self.put_connection(conn)

    def purge_knowledge_by_age(self, days: int) -> int:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => %s)",
                    (days,)
                )
                conn.commit()
                return cur.rowcount
        finally:
            self.put_connection(conn)

    def get_borderline_quality_entries(self, min_score: float = 0.6, max_score: float = 0.75) -> List[Dict]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, question, answer, q_clear, a_substance, created_at 
                    FROM knowledge 
                    WHERE (q_clear BETWEEN %s AND %s) OR (a_substance BETWEEN %s AND %s)
                    ORDER BY q_clear ASC, a_substance ASC
                    LIMIT 20
                ''', (min_score, max_score, min_score, max_score))
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        finally:
            self.put_connection(conn)

    def save_cached_response(self, question_hash: str, question: str, response: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        'SELECT id, hit_count FROM ai_response_cache WHERE question_hash = %s',
                        (question_hash,)
                    )
                    existing = cur.fetchone()

                    if existing:
                        cur.execute(
                            'UPDATE ai_response_cache SET hit_count = hit_count + 1 WHERE question_hash = %s',
                            (question_hash,)
                        )
                    else:
                        cur.execute(
                            'INSERT INTO ai_response_cache (question_hash, question, response) VALUES (%s, %s, %s)',
                            (question_hash, question, response)
                        )
                    conn.commit()
                except Exception as e:
                    print(f"Error saving cached response: {e}")
        finally:
            self.put_connection(conn)

    def get_cached_response(self, question_hash: str, max_age_days: int = 7) -> Optional[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        '''SELECT response FROM ai_response_cache 
                           WHERE question_hash = %s 
                           AND created_at > CURRENT_TIMESTAMP - make_interval(days => %s)''',
                        (question_hash, max_age_days)
                    )
                    result = cur.fetchone()

                    if result:
                        cur.execute(
                            'UPDATE ai_response_cache SET hit_count = hit_count + 1 WHERE question_hash = %s',
                            (question_hash,)
                        )
                        conn.commit()
                        return result[0]
                    return None
                except Exception as e:
                    print(f"Error getting cached response: {e}")
                    return None
        finally:
            self.put_connection(conn)

    def cleanup_old_cache(self, days: int = 7):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "DELETE FROM ai_response_cache WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => %s)",
                        (days,)
                    )
                    deleted_count = cur.rowcount
                    conn.commit()

                    if deleted_count > 0:
                        print(f"🧹 Cache cleanup: Removed {deleted_count} cached responses older than {days} days")

                    return deleted_count
                except Exception as e:
                    print(f"Error cleaning up cache: {e}")
                    return 0
        finally:
            self.put_connection(conn)

    def save_setup_message(self, channel_id: str, message_id: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM setup_messages WHERE channel_id = %s', (channel_id,))
                cur.execute(
                    'INSERT INTO setup_messages (channel_id, message_id) VALUES (%s, %s)',
                    (channel_id, message_id)
                )
                conn.commit()
        finally:
            self.put_connection(conn)

    def get_setup_message(self, channel_id: str) -> Optional[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT message_id FROM setup_messages WHERE channel_id = %s',
                    (channel_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
        finally:
            self.put_connection(conn)


db = Database()

def get_openrouter_response(messages: List[Dict], max_tokens: int = 500, max_retries: int = 3, base_delay: int = 2) -> Optional[str]:
    """Get response from OpenRouter API with exponential backoff retry logic."""
    for retry_count in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://replit.com",
                    "X-Title": "Bloom Discord Bot"
                },
                json={
                    "model": "x-ai/grok-4-fast",
                    "messages": messages,
                    "max_tokens": max_tokens
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429:
                if retry_count < max_retries - 1:
                    wait_time = base_delay * (2 ** retry_count)
                    print(f"⚠️ Rate limit hit (429). Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Max retries reached after 429 errors")
                    return "rate_limit_error"
            else:
                print(f"OpenRouter error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            print(f"⏱️ Request timed out (Attempt {retry_count + 1}/{max_retries})")
            if retry_count < max_retries - 1:
                time.sleep(base_delay)
                continue
            return "timeout_error"
        except Exception as e:
            print(f"Error getting OpenRouter response: {e}")
            if retry_count < max_retries - 1:
                time.sleep(base_delay)
                continue
            return None

    return None

def get_ai_response(prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Get AI response with improved error handling and caching."""
    cache_key = f"{system_prompt or 'default'}:{prompt}"
    cache_hash = str(deterministic_hash(cache_key))

    cached_response = db.get_cached_response(cache_hash)
    if cached_response:
        return cached_response

    messages = [
        {"role": "system", "content": system_prompt or "You are Bloom, a professional Discord bot assistant. Be direct, critical, and substantive in your responses."},
        {"role": "user", "content": prompt}
    ]

    response = get_openrouter_response(messages)

    if response == "rate_limit_error":
        return "⚠️ The AI service is currently overloaded. Please try again in a moment."
    elif response == "timeout_error":
        return "❌ Request timed out. Please try again."
    elif response:
        db.save_cached_response(cache_hash, prompt, response)
        return response
    else:
        return "❌ Sorry, I encountered an error. Please try again later."

def get_ai_response_with_history(user_id: str, question: str) -> Optional[str]:
    """Get AI response with conversation history, improved error handling, and caching."""
    history = db.get_conversation_history(user_id, 3)

    history_str = json.dumps([(conv['message'], conv['response']) for conv in history])
    cache_key = f"{history_str}:{question}"
    cache_hash = str(deterministic_hash(cache_key))

    cached_response = db.get_cached_response(cache_hash)
    if cached_response:
        db.save_conversation(user_id, question, cached_response)
        return cached_response

    messages = [
        {
            "role": "system",
            "content": "Focus on substance over praise. Skip unnecessary compliments or praise that lacks depth. Engage critically with ideas, questioning assumptions, identifying biases, and offering counterpoints where relevant. Don't shy away from disagreement when it's warranted, and ensure that any agreement is grounded in reason and evidence."
        }
    ]

    for conv in history:
        messages.append({"role": "user", "content": conv['message']})
        messages.append({"role": "assistant", "content": conv['response']})

    messages.append({"role": "user", "content": question})

    answer = get_openrouter_response(messages)

    if answer == "rate_limit_error":
        return "⚠️ The AI service is currently overloaded. Please try again in a moment."
    elif answer == "timeout_error":
        return "❌ Request timed out. Please try again."
    elif answer:
        db.save_cached_response(cache_hash, question, answer)
        db.save_conversation(user_id, question, answer)
        return answer
    else:
        return "❌ Sorry, I encountered an error. Please try again later."

async def get_legendary_roast(context: str) -> Optional[str]:
    """Generate a legendary roast with a specific prompt."""
    prompt = f"""Write a savage, comedic roast for a Discord server game feature. This is entertainment where users consent to being roasted.

Context: "{context}"

Requirements:
- Legendary, rare-event quality (0.5% trigger chance)
- Brutal, clever, memorable comedy roast
- Reference the context creatively
- 1-2 sentences max, under 300 characters
- End with a deadly emoji (💀, 🪦, ☠️, 🔥)

Examples of the style:
- "Congrats, you just unlocked Bloom's 0.5% roast… too bad your life stats are still stuck at tutorial level 💀"
- "Wow, you hit the 0.5% chance… the same odds as someone actually respecting you 🪦"
- "Lucky pull, unlucky life. Hitting this chance is the closest you'll ever get to winning anything ☠️"

Generate a devastating roast in this style:"""

    system_prompt = "You are a creative comedy writer specializing in witty roasts and insult comedy for entertainment purposes."
    return await asyncio.to_thread(get_ai_response, prompt, system_prompt)

def deterministic_hash(text: str) -> int:
    """Generate a deterministic hash using SHA256."""
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)


def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate a discriminative embedding using word-based TF-IDF style vectors."""
    if not text:
        return None

    text_lower = text.lower().strip()
    import re
    words = re.findall(r'\b\w+\b', text_lower)

    if not words:
        return None

    embedding = [0.0] * 128

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    for word, count in word_counts.items():
        word_hash = deterministic_hash(word)

        positions = []
        for i in range(min(5, len(embedding))):
            seed = abs(word_hash + i * 12345)
            positions.append(seed % len(embedding))

        tf = count / len(words)

        for pos in positions:
            embedding[pos] += tf * (1.0 + 0.1 * (word_hash % 100))

    char_bigrams = set()
    for i in range(len(text_lower) - 1):
        if text_lower[i].isalnum() and text_lower[i+1].isalnum():
            char_bigrams.add(text_lower[i:i+2])

    for bigram in sorted(char_bigrams):
        bigram_hash = deterministic_hash(bigram)
        for i in range(3):
            pos = abs(bigram_hash + i * 7919) % len(embedding)
            embedding[pos] += 0.1

    magnitude = math.sqrt(sum(x * x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    else:
        return None

    return embedding

def process_pending_conversation(thread_id: str, conversation_data: Dict):
    """Process a pending conversation after aggregation period."""
    try:
        question = conversation_data.get('question', '').strip()
        helper_messages = conversation_data.get('helper_messages', [])

        if not question or not helper_messages:
            print(f"⚠️ Skipping incomplete conversation: thread_id={thread_id}")
            return

        aggregated_answer = aggregate_messages(helper_messages)

        if not aggregated_answer:
            print(f"⚠️ No valid answer after aggregation: thread_id={thread_id}")
            return

        if is_greeting_or_casual(aggregated_answer):
            print(f"⚠️ Aggregated answer is casual/greeting, not storing: '{aggregated_answer[:50]}...'")
            return

        if not is_teaching_content(aggregated_answer):
            print(f"⚠️ Aggregated answer lacks teaching content, not storing: '{aggregated_answer[:50]}...'")
            return

        embedding = generate_embedding(question)
        if not embedding:
            print(f"⚠️ Failed to generate embedding for: '{question[:50]}...'")
            return

        saved = db.save_knowledge(question, aggregated_answer, embedding)
        if saved:
            helper_names = ', '.join(set(msg['author'] for msg in helper_messages))
            print(f"✅ AGGREGATED KB STORED | Q: '{question[:50]}...' | A: '{aggregated_answer[:50]}...' | Helpers: {helper_names}")
        else:
            print(f"⚠️ Failed to store aggregated KB: '{question[:50]}...'")

    except Exception as e:
        print(f"❌ Error processing conversation {thread_id}: {e}")

async def process_expired_conversations():
    """Background task to process conversations after 10-minute window."""
    global pending_conversations

    while True:
        try:
            await asyncio.sleep(30)

            current_time = datetime.now(timezone.utc)
            expired_threads = []

            with conversation_lock:
                for thread_id, conv_data in list(pending_conversations.items()):
                    last_message_time = conv_data.get('last_update')
                    if last_message_time:
                        time_elapsed = (current_time - last_message_time).total_seconds()

                        if time_elapsed >= 600:
                            expired_threads.append((thread_id, conv_data))
                            del pending_conversations[thread_id]

            for thread_id, conv_data in expired_threads:
                await asyncio.to_thread(process_pending_conversation, thread_id, conv_data)

        except Exception as e:
            print(f"❌ Error in background conversation processor: {e}")

ROLE_IDS = {
    'Rainbow': '1423477410409746695',
    'Aurora Borealis': '1423477508376100945',
    'Eclipse': '1423477456610000966',
    'Starfall': '1423477374183542874',
    'Admin Event': '1425951321130926296',
    'Shiny Surge': '1426295947482108056',
    'Mutation Surge': '1426296058773770423',
    'Night of the Fireflies': '1426296253263646720',
    'Night of the Luminous': '1426296472210767992'
}

class RoleSelectView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.select(
        placeholder="Choose roles to get pinged for...",
        min_values=0,
        max_values=len(ROLE_IDS),
        options=[
            discord.SelectOption(label="Rainbow", value=ROLE_IDS["Rainbow"], emoji="<:Rainbow1:1426295065134891058>"),
            discord.SelectOption(label="Aurora Borealis", value=ROLE_IDS["Aurora Borealis"], emoji="<:Aurora_Borealis:1426295002660601926>"),
            discord.SelectOption(label="Eclipse", value=ROLE_IDS["Eclipse"], emoji="<:Eclipse1:1426295080087457882>"),
            discord.SelectOption(label="Starfall", value=ROLE_IDS["Starfall"], emoji="<:Starfall:1426295049431552171>"),
            discord.SelectOption(label="Admin Event", value=ROLE_IDS["Admin Event"], emoji="👿"),
            discord.SelectOption(label="Shiny Surge", value=ROLE_IDS["Shiny Surge"], emoji="<:ShinySurge:1426297213490958458>"),
            discord.SelectOption(label="Mutation Surge", value=ROLE_IDS["Mutation Surge"], emoji="<:MutationSurge:1426297232776368188>"),
            discord.SelectOption(label="Night of the Fireflies", value=ROLE_IDS["Night of the Fireflies"], emoji="<:NightoftheFireflies:1426297251004944454>"),
            discord.SelectOption(label="Night of the Luminous", value=ROLE_IDS["Night of the Luminous"], emoji="<:NightoftheLuminous:1426297267857526844>")
        ],
        custom_id="role_select"
    )
    async def select_callback(self, interaction: discord.Interaction, select: discord.ui.Select):
        await interaction.response.defer(ephemeral=True)
        
        selected_role_ids = set(select.values)
        current_role_ids = {str(role.id) for role in interaction.user.roles}
        
        roles_to_add = []
        roles_to_remove = []
        
        for role_name, role_id in ROLE_IDS.items():
            role = interaction.guild.get_role(int(role_id))
            if not role:
                continue
                
            if role_id in selected_role_ids:
                if role_id not in current_role_ids:
                    roles_to_add.append(role)
            else:
                if role_id in current_role_ids:
                    roles_to_remove.append(role)
        
        if roles_to_add:
            await interaction.user.add_roles(*roles_to_add, reason="Role selection via /setup")
        if roles_to_remove:
            await interaction.user.remove_roles(*roles_to_remove, reason="Role deselection via /setup")
        
        added_names = [role.name for role in roles_to_add]
        removed_names = [role.name for role in roles_to_remove]
        
        message_parts = []
        if added_names:
            message_parts.append(f"✅ Added: {', '.join(added_names)}")
        if removed_names:
            message_parts.append(f"❌ Removed: {', '.join(removed_names)}")
        
        if message_parts:
            await interaction.followup.send('\n'.join(message_parts), ephemeral=True)
        else:
            await interaction.followup.send("No changes made to your roles.", ephemeral=True)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    
    bot.add_view(RoleSelectView())
    print("✅ Persistent role selection view registered")
    
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Error syncing commands: {e}')

    bot.loop.create_task(process_expired_conversations())
    print("🔄 Started background conversation aggregation processor (10-minute window)")

@bot.event
async def on_member_join(member):
    account_age = datetime.now(timezone.utc) - member.created_at

    if account_age.days < 7:
        try:
            await member.send(
                f"Hello {member.name},\n\n"
                f"Your account is less than 7 days old. For security reasons, "
                f"you have been automatically kicked from the server.\n\n"
                f"You are welcome to rejoin once your account is at least 7 days old."
            )
        except:
            pass

        try:
            await member.kick(reason="Account < 7 days old")

            auto_kick_channel = bot.get_channel(int(AUTO_KICK_CHANNEL))
            if auto_kick_channel:
                kick_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                await auto_kick_channel.send(
                    f"**Moderation Action:** kick\n"
                    f"**Target:** <@{member.id}> (ID: {member.id})\n"
                    f"**By:** Bloom (auto-kick)\n"
                    f"**Reason:** Account < 7 days old\n"
                    f"**Time:** {kick_time}"
                )
        except Exception as e:
            print(f"Error kicking member: {e}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    global roast_chance
    if random.random() < (roast_chance / 100):
        try:
            context = f"{message.author.name} said: {message.content}"
            roast = await get_legendary_roast(context)
            if roast:
                await message.reply(roast)
                print(f"🔥 LEGENDARY ROAST TRIGGERED ({roast_chance}% chance) | Target: {message.author.name}")
        except Exception as e:
            print(f"Error generating roast: {e}")

    if str(message.channel.id) == TRADING_CHANNEL:
        user_id = str(message.author.id)
        content = message.content.strip().lower()

        if user_id not in user_messages:
            user_messages[user_id] = []

        user_message_timestamps[user_id] = datetime.now(timezone.utc)

        if content in user_messages[user_id]:
            try:
                await message.delete()
                await message.author.timeout(timedelta(minutes=10), reason="Repeated trade message in trading channel")

                temp_msg = await message.channel.send(
                    f"{message.author.mention} You have been muted for 10 minutes for repeating your trade message. "
                    f"Please remember:\n"
                    f"• Do not repeat your trade more than once\n"
                    f"• Maximum 7 lines per trade\n"
                    f"• Take conversations to DMs"
                )
                await temp_msg.delete(delay=10)
                print(f"Muted {message.author.name} for repeating message in trading channel")
            except Exception as e:
                print(f"Error muting user: {e}")
        else:
            user_messages[user_id].append(content)
            if len(user_messages[user_id]) > 10:
                user_messages[user_id].pop(0)

        await bot.process_commands(message)
        return

    if str(message.channel.id) == LEARNING_CHANNEL:
        member = message.author
        has_helper_role = any(str(role.id) in HELPER_ROLES for role in member.roles)
        has_student_role = any(str(role.id) == STUDENT_ROLE for role in member.roles)

        if message.reference and has_helper_role:
            try:
                referenced_msg = await message.channel.fetch_message(message.reference.message_id)

                if referenced_msg and not referenced_msg.author.bot:
                    student_has_role = any(str(role.id) == STUDENT_ROLE for role in referenced_msg.author.roles)

                    if student_has_role:
                        question = referenced_msg.content.strip()
                        answer_content = message.content.strip()

                        if question and answer_content:
                            if is_blabberish(answer_content):
                                print(f"⚠️ Detected blabberish from {message.author.name}, ignoring: '{answer_content[:30]}...'")
                                await bot.process_commands(message)
                                return

                            thread_id = f"{referenced_msg.id}"

                            with conversation_lock:
                                if thread_id not in pending_conversations:
                                    pending_conversations[thread_id] = {
                                        'question': question,
                                        'helper_messages': [],
                                        'last_update': datetime.now(timezone.utc),
                                        'student_id': str(referenced_msg.author.id)
                                    }

                                pending_conversations[thread_id]['helper_messages'].append({
                                    'content': answer_content,
                                    'author': message.author.name,
                                    'timestamp': datetime.now(timezone.utc)
                                })
                                pending_conversations[thread_id]['last_update'] = datetime.now(timezone.utc)

                                msg_count = len(pending_conversations[thread_id]['helper_messages'])
                                print(f"📝 Helper message tracked ({msg_count} total) | Thread: {thread_id} | Helper: {message.author.name} | Msg: '{answer_content[:40]}...'")
            except Exception as e:
                print(f"Error in learning system: {e}")

        elif has_student_role and not message.reference:
            try:
                question = message.content.strip()

                if question:
                    question_embedding = generate_embedding(question)

                    if question_embedding:
                        top_matches = db.find_top_matches_with_confidence(question, question_embedding, top_n=3, min_semantic=0.65)

                        if top_matches:
                            best_match = top_matches[0]
                            combined_conf = best_match['combined_confidence']

                            if combined_conf >= KB_REPLY_CONFIDENCE:
                                rephrase_prompt = f"Helper's answer: {best_match['answer']}\n\nStudent's question: {question}"
                                rephrased = await asyncio.to_thread(get_ai_response, rephrase_prompt, KB_REPLY_MODE)

                                if rephrased:
                                    confidence_percent = int(combined_conf * 100)
                                    await message.reply(
                                        f"{rephrased}\n\n||[KB match - {confidence_percent}% confidence]||"
                                    )
                                    print(f"📚 KB Reply (rephrased) | Confidence: {confidence_percent}% | Question: '{question[:50]}...'")

                            elif combined_conf >= KB_SUGGEST_CONFIDENCE:
                                confidence_percent = int(combined_conf * 100)
                                await message.reply(
                                    f"{best_match['answer']}\n\n||[KB match - {confidence_percent}% confidence]||"
                                )
                                print(f"📚 KB Suggest | Confidence: {confidence_percent}% | Question: '{question[:50]}...'")

                            else:
                                print(f"🔇 Silent (confidence too low: {combined_conf:.2f}) | Question: '{question[:50]}...'")
                        else:
                            print(f"🔇 Silent (no KB match) | Question: '{question[:50]}...'")
            except Exception as e:
                print(f"Error in auto-reply system: {e}")

    await bot.process_commands(message)

@bot.tree.command(name="say", description="Make the bot say something")
@app_commands.describe(message="The message to say")
async def say(interaction: discord.Interaction, message: str):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    await interaction.channel.send(message)
    await interaction.followup.send("Message sent!", ephemeral=True)

@bot.tree.command(name="vibe", description="Get a vibe check from Bloom")
async def vibe(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    vibes = [
        "The vibes are immaculate ✨",
        "Vibes are passing, could be better 🤔",
        "Vibes are kinda off ngl 😬",
        "Straight up not having a good time rn 💀",
        "The energy is unmatched today 🔥",
        "Mid vibes, nothing special 😐",
        "Vibes are questionable at best 👀",
        "Peak vibes achieved 🎯"
    ]
    await interaction.response.send_message(random.choice(vibes))

@bot.tree.command(name="quote", description="Get a motivational quote")
async def quote(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    quotes = [
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "The only way to do great work is to love what you do.",
        "Believe you can and you're halfway there.",
        "Don't watch the clock; do what it does. Keep going.",
        "The future belongs to those who believe in the beauty of their dreams.",
        "It does not matter how slowly you go as long as you do not stop.",
        "Everything you've ever wanted is on the other side of fear.",
        "Success is walking from failure to failure with no loss of enthusiasm.",
        "The only impossible journey is the one you never begin.",
        "In the middle of difficulty lies opportunity."
    ]
    await interaction.response.send_message(f"💭 *{random.choice(quotes)}*")

@bot.tree.command(name="8ball", description="Ask the magic 8-ball a question")
@app_commands.describe(question="Your yes/no question")
async def eightball(interaction: discord.Interaction, question: str):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    responses = [
        "Yes, absolutely.",
        "No doubt about it.",
        "Maybe, maybe not.",
        "Definitely not.",
        "Ask again later.",
        "I wouldn't count on it.",
        "The signs point to yes.",
        "Better not tell you now.",
        "Outlook not so good.",
        "It is certain.",
        "Very doubtful.",
        "Without a doubt.",
        "My sources say no.",
        "As I see it, yes.",
        "Cannot predict now."
    ]
    await interaction.response.send_message(f"🎱 {random.choice(responses)}")

@bot.tree.command(name="tellmeajoke", description="Get a joke from Bloom")
@app_commands.describe(
    context="Optional context for the joke",
    user="Optional user to mention"
)
async def tellmeajoke(interaction: discord.Interaction, context: Optional[str] = None, user: Optional[discord.User] = None):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    jokes = [
        "Why don't scientists trust atoms? Because they make up everything.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why don't skeletons fight each other? They don't have the guts.",
        "What do you call fake spaghetti? An impasta.",
        "I'm reading a book about anti-gravity. It's impossible to put down.",
        "Why did the scarecrow win an award? He was outstanding in his field.",
        "I used to hate facial hair, but then it grew on me.",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
        "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
        "Why do programmers prefer dark mode? Because light attracts bugs."
    ]

    if context:
        prompt = f"Tell a short roasting joke about: {context}. Be clever and edgy. 1-2 sentences max."
        system_prompt = "You are a roast comedy bot. Make sharp, witty jokes. Keep it short and brutal."
        joke = get_ai_response(prompt, system_prompt)
    else:
        joke = random.choice(jokes)

    if joke:
        final_joke = f"{user.mention} {joke}" if user else joke
        await interaction.followup.send(final_joke)
    else:
        await interaction.followup.send("Failed to generate joke.")

@bot.tree.command(name="askbloom", description="Ask Bloom anything")
@app_commands.describe(question="Your question")
async def askbloom(interaction: discord.Interaction, question: str):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    answer = get_ai_response_with_history(str(interaction.user.id), question)

    if answer:
        await interaction.followup.send(answer)
    else:
        await interaction.followup.send("Sorry, I encountered an error processing your question.")

@bot.tree.command(name="ban", description="Ban a user (Owner only)")
@app_commands.describe(
    user="The user to ban",
    reason="Reason for the ban"
)
async def ban(interaction: discord.Interaction, user: discord.Member, reason: str):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    try:
        await user.ban(reason=reason)
        await interaction.followup.send(f"Successfully banned {user.mention} for: {reason}", ephemeral=True)

        auto_kick_channel = bot.get_channel(int(AUTO_KICK_CHANNEL))
        if auto_kick_channel:
            ban_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            await auto_kick_channel.send(
                f"**Moderation Action:** ban\n"
                f"**Target:** <@{user.id}> (ID: {user.id})\n"
                f"**By:** {interaction.user.mention}\n"
                f"**Reason:** {reason}\n"
                f"**Time:** {ban_time}"
            )
    except Exception as e:
        await interaction.followup.send(f"Failed to ban user: {str(e)}", ephemeral=True)

@bot.tree.command(name="roastchance", description="Set the legendary roast trigger chance percentage")
@app_commands.describe(percentage="The percentage chance (0.0 to 100.0) for Bloom to roast")
async def roastchance(interaction: discord.Interaction, percentage: float):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    if percentage < 0 or percentage > 100:
        await interaction.response.send_message("Percentage must be between 0.0 and 100.0.", ephemeral=True)
        return

    global roast_chance
    old_chance = roast_chance
    roast_chance = percentage

    await interaction.response.send_message(
        f"🔥 Legendary roast chance updated!\n"
        f"**Old:** {old_chance}%\n"
        f"**New:** {roast_chance}%\n\n"
        f"*Every message now has a {roast_chance}% chance of triggering a devastating roast.*",
        ephemeral=True
    )
    print(f"⚙️ Roast chance updated: {old_chance}% → {roast_chance}% (by {interaction.user.name})")

@bot.tree.command(name="kbstats", description="Show knowledge base statistics")
async def kbstats(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    stats = db.get_database_stats()

    embed = discord.Embed(
        title="📊 Knowledge Base Statistics",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="📚 Knowledge Base",
        value=(
            f"**Total Entries:** {stats.get('knowledge_count', 0)}\n"
            f"**Avg Question Clarity:** {stats.get('avg_q_clear', 0):.2f}\n"
            f"**Avg Answer Substance:** {stats.get('avg_a_substance', 0):.2f}\n"
            f"**High Quality (≥0.8):** {stats.get('high_q_clear', 0)} Q / {stats.get('high_a_substance', 0)} A\n"
            f"**Medium Quality (0.6-0.8):** {stats.get('medium_q_clear', 0)} Q / {stats.get('medium_a_substance', 0)} A"
        ),
        inline=False
    )

    embed.add_field(
        name="💬 Conversations",
        value=(
            f"**Total Conversations:** {stats.get('conversations_count', 0)}\n"
            f"**Unique Users:** {stats.get('unique_users', 0)}"
        ),
        inline=False
    )

    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="kbexport", description="Export knowledge base as JSON")
async def kbexport(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    json_data = db.export_knowledge_as_json()

    if len(json_data) > 1900:
        await interaction.followup.send(
            f"📦 Knowledge base exported!\n\n**Entry count:** {db.get_database_stats().get('knowledge_count', 0)}\n\n"
            f"*Data is too large to display. Use the FastAPI endpoint `/kbexport` or database tools to access full export.*",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"📦 Knowledge base exported:\n```json\n{json_data}\n```",
            ephemeral=True
        )

@bot.tree.command(name="extractkb", description="Extract knowledge base as JSON (alias of /kbexport)")
async def extractkb(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    json_data = db.export_knowledge_as_json()

    if len(json_data) > 1900:
        await interaction.followup.send(
            f"📦 Knowledge base exported:\n\n**Entry count:** {db.get_database_stats().get('knowledge_count', 0)}\n\n"
            f"*Data is too large to display. Use the FastAPI endpoint `/kbexport` or database tools to access full export.*",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"📦 Knowledge base exported:\n```json\n{json_data}\n```",
            ephemeral=True
        )

@bot.tree.command(name="kbpurge", description="Remove knowledge base entries")
@app_commands.describe(
    entry_id="Entry ID to remove (optional)",
    days_old="Remove entries older than X days (optional)"
)
async def kbpurge(interaction: discord.Interaction, entry_id: Optional[int] = None, days_old: Optional[int] = None):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    if entry_id is not None:
        success = db.purge_knowledge_by_id(entry_id)
        if success:
            await interaction.followup.send(f"✅ Successfully removed entry ID {entry_id}", ephemeral=True)
            print(f"🗑️ KB entry {entry_id} purged by {interaction.user.name}")
        else:
            await interaction.followup.send(f"❌ Entry ID {entry_id} not found", ephemeral=True)

    elif days_old is not None:
        count = db.purge_knowledge_by_age(days_old)
        await interaction.followup.send(f"✅ Removed {count} entries older than {days_old} days", ephemeral=True)
        print(f"🗑️ {count} KB entries purged (>{days_old} days old) by {interaction.user.name}")

    else:
        await interaction.followup.send("❌ Please specify either entry_id or days_old", ephemeral=True)

@bot.tree.command(name="kbreview", description="Review borderline quality knowledge base entries")
async def kbreview(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    entries = db.get_borderline_quality_entries(min_score=0.6, max_score=0.75)

    if not entries:
        await interaction.followup.send("✅ No borderline quality entries found!", ephemeral=True)
        return

    embed = discord.Embed(
        title="⚠️ Borderline Quality Entries",
        description=f"Found {len(entries)} entries with quality scores between 0.6-0.75",
        color=discord.Color.orange()
    )

    for entry in entries[:5]:
        embed.add_field(
            name=f"ID {entry['id']} | Q:{entry.get('q_clear', 0):.2f} A:{entry.get('a_substance', 0):.2f}",
            value=f"**Q:** {entry['question'][:80]}{'...' if len(entry['question']) > 80 else ''}\n"
                  f"**A:** {entry['answer'][:80]}{'...' if len(entry['answer']) > 80 else ''}",
            inline=False
        )

    if len(entries) > 5:
        embed.set_footer(text=f"Showing 5 of {len(entries)} entries. Use /kbpurge to remove low-quality entries.")

    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="saywb", description="Send an embedded message with optional title and color")
@app_commands.describe(
    description="The message description (required)",
    title="Optional title for the embed",
    color="Optional color (gray, red, pink, blue, green, yellow, purple, orange)"
)
async def saywb(interaction: discord.Interaction, description: str, title: Optional[str] = None, color: Optional[str] = None):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    # Map color names to Discord colors
    color_map = {
        'gray': discord.Color.light_gray(),
        'grey': discord.Color.light_gray(),
        'red': discord.Color.red(),
        'pink': discord.Color.pink(),
        'blue': discord.Color.blue(),
        'green': discord.Color.green(),
        'yellow': discord.Color.yellow(),
        'purple': discord.Color.purple(),
        'orange': discord.Color.orange()
    }

    # Get the color, default to black (dark gray)
    embed_color = color_map.get(color.lower(), discord.Color.from_rgb(0, 0, 0)) if color else discord.Color.from_rgb(0, 0, 0)

    # Create the embed
    if title:
        embed = discord.Embed(title=title, description=description, color=embed_color)
    else:
        embed = discord.Embed(description=description, color=embed_color)

    await interaction.channel.send(embed=embed)
    await interaction.followup.send("Embed sent!", ephemeral=True)

@bot.tree.command(name="setup", description="Setup role selection message")
async def setup(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    existing_message_id = db.get_setup_message(str(interaction.channel_id))
    if existing_message_id:
        try:
            existing_message = await interaction.channel.fetch_message(int(existing_message_id))
            await existing_message.delete()
        except:
            pass

    embed = discord.Embed(
        title="Click the button and choose a role on what u wanna get pinged on",
        description="Select the roles you want to be pinged for. Selecting them again will remove them.",
        color=discord.Color.pink()
    )

    view = RoleSelectView()
    message = await interaction.channel.send(embed=embed, view=view)
    
    db.save_setup_message(str(interaction.channel_id), str(message.id))
    
    await interaction.followup.send("✅ Role selection message has been set up!", ephemeral=True)
    print(f"🎯 Setup message created in channel {interaction.channel_id} by {interaction.user.name}")

@bot.tree.command(name="help", description="Show all available commands")
async def help_command(interaction: discord.Interaction):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    embed = discord.Embed(
        title="🌸 Bloom Bot - Command List",
        description="Here are all the commands you can use with Bloom:",
        color=discord.Color.pink()
    )

    embed.add_field(
        name="🤖 AI Features",
        value=(
            "**`/askbloom`** - Ask Bloom anything\n"
            "**`/tellmeajoke`** - Get a joke\n"
        ),
        inline=False
    )

    embed.add_field(
        name="🎉 Fun Commands",
        value=(
            "**`/say`** - Make the bot say something\n"
            "**`/saywb`** - Send an embedded message with optional title and color\n"
            "**`/vibe`** - Get a vibe check from Bloom\n"
            "**`/quote`** - Get a motivational quote\n"
            "**`/8ball`** - Ask the magic 8-ball a question\n"
        ),
        inline=False
    )

    embed.add_field(
        name="🛡️ Moderation (Owner Only)",
        value=(
            "**`/ban`** - Ban a user from the server\n"
            "**`/roastchance`** - Set the legendary roast trigger percentage\n"
            "**`/setup`** - Setup role selection message with persistent buttons\n"
        ),
        inline=False
    )

    embed.add_field(
        name="📊 Knowledge Base Admin (Owner Only)",
        value=(
            "**`/kbstats`** - View KB statistics and quality metrics\n"
            "**`/kbexport`** - Export KB as JSON\n"
            "**`/extractkb`** - Extract KB as JSON\n"
            "**`/export`** - Export KB as JSON\n"
            "**`/kbpurge`** - Remove entries by ID or age\n"
            "**`/kbreview`** - Review borderline quality entries\n"
        ),
        inline=False
    )

    embed.set_footer(text="Bloom Bot | Use commands to interact!")

    await interaction.response.send_message(embed=embed, ephemeral=True)

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    from threading import Thread

    app = FastAPI()

    @app.get("/")
    def home():
        return {"status": "ok", "bot": "Bloom", "message": "Bot is running"}

    @app.get("/health")
    def health():
        return {"status": "ok", "bot": "Bloom", "active": True}

    @app.get("/kbstats")
    def get_kb_stats():
        stats = db.get_database_stats()
        return {
            "knowledge_entries": stats.get('knowledge_count', 0),
            "total_conversations": stats.get('conversations_count', 0),
            "unique_users": stats.get('unique_users', 0)
        }

    @app.get("/kbsearch")
    def kbsearch(q: str):
        question_embedding = generate_embedding(q)
        if not question_embedding:
            return {"error": "Could not generate embedding for query"}

        knowledge = db.get_all_knowledge()
        results = []

        for item in knowledge:
            stored_embedding = json.loads(item['embedding'])
            similarity = db.cosine_similarity(question_embedding, stored_embedding)

            if similarity >= 0.5:
                results.append({
                    "question": item['question'],
                    "answer": item['answer'],
                    "similarity": float(similarity),
                    "created_at": item['created_at']
                })

        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
        return {"results": results, "count": len(results)}

    def run_fastapi():
        uvicorn.run(app, host='0.0.0.0', port=PORT, log_level="info")

    # Start FastAPI in a separate thread
    Thread(target=run_fastapi, daemon=True).start()

    # Run the Discord bot
    print(f"🚀 Starting Bloom bot with FastAPI server on port {PORT}")
    bot.run(TOKEN)
