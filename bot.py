import os
import sqlite3
import json
import math
import requests
import hashlib
import random
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import discord
from discord import app_commands
from discord.ext import commands
LEARNING_CHANNEL = '1411335494234669076'
TRADING_CHANNEL = '1418976581099065355'
AUTO_KICK_CHANNEL = '1411335541873709167'
HELPER_ROLES = ['1418434355650625676', '1352853011424219158', '1372300233240739920']
STUDENT_ROLE = '1341949236471926805'
OWNER_ID = '1334138321412296725'
PORT = int(os.getenv('PORT', 8080))

TOKEN = os.getenv('TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

user_messages = {}
roast_chance = 0.5

class Database:
    def __init__(self, db_path='bloom.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_knowledge(self, question: str, answer: str, embedding: List[float]) -> bool:
        try:
            cursor = self.conn.execute(
                'INSERT OR IGNORE INTO knowledge (question, answer, embedding) VALUES (?, ?, ?)',
                (question, answer, json.dumps(embedding))
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error saving knowledge: {e}")
            return False
    
    def get_all_knowledge(self) -> List[Dict]:
        cursor = self.conn.execute('SELECT * FROM knowledge')
        return [dict(row) for row in cursor.fetchall()]
    
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
    
    def save_conversation(self, user_id: str, message: str, response: str):
        self.conn.execute(
            'INSERT INTO conversations (user_id, message, response) VALUES (?, ?, ?)',
            (user_id, message, response)
        )
        self.conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        cursor = self.conn.execute(
            'SELECT message, response FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
            (user_id, limit)
        )
        return list(reversed([dict(row) for row in cursor.fetchall()]))

db = Database()

def get_openrouter_response(messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
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
                "model": "x-ai/grok-4-fast:free",
                "messages": messages,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"OpenRouter error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting OpenRouter response: {e}")
        return None

def get_ai_response(prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    messages = [
        {"role": "system", "content": system_prompt or "You are Bloom, a helpful Discord bot assistant."},
        {"role": "user", "content": prompt}
    ]
    return get_openrouter_response(messages)

def get_ai_response_with_history(user_id: str, question: str) -> Optional[str]:
    history = db.get_conversation_history(user_id, 5)
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
    if answer:
        db.save_conversation(user_id, question, answer)
    return answer

def get_legendary_roast(context: str) -> Optional[str]:
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
    return get_ai_response(prompt, system_prompt)

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
    
    embedding = [0.0] * 384
    
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

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Error syncing commands: {e}')

@bot.event
async def on_member_join(member):
    from datetime import datetime, timezone
    
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
            roast = get_legendary_roast(context)
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
                        answer = message.content.strip()
                        
                        if question and answer:
                            embedding = generate_embedding(question)
                            
                            if embedding:
                                saved = db.save_knowledge(question, answer, embedding)
                                if saved:
                                    print(f"✅ KV STORED | Question: '{question[:50]}...' | Answer: '{answer[:50]}...' | Helper: {message.author.name}")
                                else:
                                    print(f"⚠️ Duplicate question detected, not stored: '{question[:50]}...'")
            except Exception as e:
                print(f"Error in learning system: {e}")
        
        elif has_student_role and not message.reference:
            try:
                question = message.content.strip()
                
                if question:
                    question_embedding = generate_embedding(question)
                    
                    if question_embedding:
                        match = db.find_similar_question(question_embedding, threshold=0.65)
                        
                        if match:
                            similarity_percent = int(match['similarity'] * 100)
                            await message.reply(
                                f"{match['answer']}\n\n||[Bloom learned this from helpers so it might not be correct and dont be shy to ping helpers to solve ur answer thank you]|| - {similarity_percent}% confidence"
                            )
                            print(f"📚 Answered from knowledge base | Similarity: {similarity_percent}% | Question: '{question[:50]}...'")
            except Exception as e:
                print(f"Error in auto-reply system: {e}")
    
    await bot.process_commands(message)

@bot.tree.command(name="say", description="Make the bot say something")
@app_commands.describe(message="The message to say")
async def say(interaction: discord.Interaction, message: str):
    await interaction.response.defer(ephemeral=True)
    await interaction.channel.send(message)
    await interaction.followup.send("Message sent!", ephemeral=True)

@bot.tree.command(name="tellmeajoke", description="Get a joke from Bloom")
@app_commands.describe(
    context="Optional context for the joke",
    user="Optional user to mention"
)
async def tellmeajoke(interaction: discord.Interaction, context: Optional[str] = None, user: Optional[discord.User] = None):
    await interaction.response.defer(ephemeral=True)
    
    if context:
        prompt = f"Tell me a unique joke about: {context}. It can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke."
    else:
        prompt = "Tell me a unique joke. It can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke."
    
    system_prompt = "You are a humor bot. When this command is used, respond with a joke. The joke can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke. Each response should be unique and not a repeat of the last one."
    
    joke = get_ai_response(prompt, system_prompt)
    
    if joke:
        final_joke = f"{user.mention} {joke}" if user else joke
        await interaction.channel.send(final_joke)
        await interaction.followup.send("Joke sent!", ephemeral=True)
    else:
        await interaction.followup.send("Failed to generate joke.", ephemeral=True)

@bot.tree.command(name="askbloom", description="Ask Bloom anything")
@app_commands.describe(question="Your question")
async def askbloom(interaction: discord.Interaction, question: str):
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
            from datetime import datetime, timezone
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

if __name__ == "__main__":
    from flask import Flask
    from threading import Thread
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "Bloom Bot is running!"
    
    def run_flask():
        app.run(host='0.0.0.0', port=PORT)
    
    # Start Flask in a separate thread
    Thread(target=run_flask).start()
    
    # Run the Discord bot
    bot.run(TOKEN)
